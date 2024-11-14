# -*- origami-fold-style: triple-braces; coding: utf-8; -*-
# MFTIQ - WACV2025
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '' # has to be set before torch is imported, otherwise it is run on wrong device (error is wierd, sice occurs only sometimes)

# sys.insert(0, '/home.stud/neoramic/repos/mft2024_debug')

import argparse
from pathlib import Path
import logging
import warnings
import datetime

import numpy as np
import torch
from torch import nn

from rich.progress import Progress

from MFTIQ.UOM.network.configs import UOMConfigs
from MFTIQ.UOM.datasets import datasets
from MFTIQ.RAFT.train import (
    count_parameters, Timer, fetch_optimizer, Logger,
    GradScaler, sequence_loss
)

from MFTIQ.UOM.datasets.evaluate import validate_sintel, validate_kubric, validate_kubric_2024
from MFTIQ.UOM.losses.uncertainty_threshold_loss import unc_thr_loss
from MFTIQ.UOM.losses.loss_and_prediction import universal_prediction_fce
from MFTIQ.UOM.datasets.augmentations import Rotate90Augmentor

# do not remove, needed for creation of dataset cache files
from MFTIQ.UOM.create_kubric_data import CompStatus

logger = logging.getLogger(__name__)

# exclude extremely large displacements
MAX_FLOW = 760
SUM_FREQ = 100
VAL_FREQ = 5000
VAL_FREQ_DEBUG = 100
FIRST_VAL_STEP = 7


def parse_arguments():
    parser = argparse.ArgumentParser(description='Training of standalone Uncertainty Occlusion Module', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', help='', action='store_true')

    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help='Cuda devices')
    parser.add_argument('--config', required=True, type=str, help='config setting')
    parser.add_argument('--logdir', default=Path('runs'), type=Path, help='logging directory for tensorboard')
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--checkpoints', type=Path, help='checkpoint directory', default=Path('./checkpoints'))
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--debug', help='debug mode (disable progressbars, etc)', action='store_true')
    parser.add_argument('--n_workers', type=int, default=8, help='dataloader workers')

    args = parser.parse_args()
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu_n) for gpu_n in args.gpus])

    format = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    lvl = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=lvl, format=format) 
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    return args

def is_validation_step(total_steps, debug=False):
    if (total_steps == FIRST_VAL_STEP) or (total_steps % VAL_FREQ == VAL_FREQ - 1):
        return True
    elif debug and (total_steps % VAL_FREQ_DEBUG == VAL_FREQ_DEBUG - 1):
        return True
    else:
        return False

def train(args):

    train_timer = Timer()

    conf = UOMConfigs(args)
    conf.print_config()

    training_args = conf.get_train_args()
    args.gpus = range(len(args.gpus))
    uom_args = conf.get_uom_args()

    # LOAD MODEL
    with warnings.catch_warnings():
        # catch warning from DINOv2 because guys from
        # facebookresearch are misusing warnings to inform about
        # success...
        warnings.filterwarnings(action='ignore',
                                category=UserWarning,
                                message=r'xFormers is available.*')
        model = conf.model(**uom_args)
    model = nn.DataParallel(model, device_ids=args.gpus)
    model = model.module

    print("Parameter Count: %d" % count_parameters(model))

    if training_args.restore_checkpoint is not None:
        model.load_state_dict(torch.load(training_args.restore_checkpoint), strict=False)

    model.cuda()
    model.train()

    # TRAINING METHODS
    train_loader = datasets.fetch_dataloader(training_args)
    optimizer, scheduler = fetch_optimizer(training_args, model)

    # LOAD FLOW METHOD
    flow_args = conf.get_flow_args()
    flow_estimator = conf.flow_method(flow_args)
    flow_estimator = nn.DataParallel(flow_estimator, device_ids=args.gpus)
    flow_estimator.load_state_dict(torch.load(flow_args.checkpoint, map_location='cpu'))
    flow_estimator = flow_estimator.module
    flow_estimator.cuda()
    flow_estimator.eval()
    flow_estimator.requires_grad_(False)

    total_steps = 0
    scaler = GradScaler(enabled=training_args.mixed_precision)
    stamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    logdir = args.logdir / f'{stamp}--{conf.config_name}'
    logger = Logger(model, scheduler, logdir=logdir)

    should_keep_training = True
    print('Training...', train_timer.iter(), train_timer())

    with Progress(disable=args.debug) as progress:
        training_task = progress.add_task("[red]Training...", total=training_args.num_steps)

        while should_keep_training:
            for i_batch, data_blob in enumerate(train_loader):

                augm_rotate90_enabled = training_args.augmentation_rotate90 and np.random.uniform(0.0, 1.0) > 0.33
                augm_rotate90 = Rotate90Augmentor(augm_rotate90_enabled)

                if is_validation_step(total_steps, debug=args.debug):
                    model.eval()

                optimizer.zero_grad()
                c_data = {k:v.cuda() for k,v in data_blob.items()}
                image2_for_flow_est = c_data['img2']
                c_data = augm_rotate90(c_data)

                image1, image2, flow_gt = c_data['img1'], c_data['img2'], c_data['flow']
                valid_gt, occl_gt = c_data['valid'], c_data['occl']


                if training_args.add_noise:
                    stdv = np.random.uniform(0.0, 5.0)
                    image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                    image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)
                    image2_for_flow_est = (image2_for_flow_est + stdv * torch.randn(*image2_for_flow_est.shape).cuda()).clamp(0.0, 255.0)

                flow_pick_prob = np.random.uniform(0.0, 1.0)
                if training_args.random_pick_flow and flow_pick_prob > 0.33:
                    if flow_pick_prob > 0.66:
                        # flow (FlowFormer) pre-computed on whole images
                        flow_est = c_data['flow_est']
                    else:
                        # ground truth flow
                        flow_est = torch.clip(c_data['flow'], -1000.0, 1000.0)
                else:
                    # flow (RAFT) computed on-the-fly on the augmented, cropped data
                    flow_est_dict = flow_estimator(image1, image2_for_flow_est, test_mode=True)
                    flow_est_dict = augm_rotate90(flow_est_dict)
                    flow_est = flow_est_dict['flow']

                all_predictions = model(image1, image2, flow=flow_est)

                for k in all_predictions.keys():
                    all_predictions[k] = [all_predictions[k]]

                if training_args.uncertainty_loss != 'sigma':
                    loss, metrics = unc_thr_loss(all_predictions, flow_gt, valid_gt, occl_gt=occl_gt, gamma=training_args.gamma, args=training_args,
                                                 occlusion_loss_type=training_args.occlusion_loss,
                                                 uncertainty_loss_type=training_args.uncertainty_loss)
                else:
                    assert training_args.occlusion_loss == 'cross_entropy'
                    assert training_args.uncertainty_loss == 'sigma'
                    loss, metrics = sequence_loss(all_predictions, flow_gt, valid_gt, occl_gt=occl_gt, gamma=training_args.gamma, args=training_args)

                if not torch.all(torch.isfinite(loss)):
                    print('loss is not finite')
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_args.clip)

                scaler.step(optimizer)
                scheduler.step()
                scaler.update()

                logger.push(metrics)

                if is_validation_step(total_steps, debug=args.debug):
                    print('validation ', i_batch, total_steps, train_timer.iter(), train_timer())
                    PATH = args.checkpoints / f'{total_steps+1:06d}_{args.name}.pth'
                    PATH.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), PATH)

                    results = {}

                    validation_task = progress.add_task("[blue]Validation...", total=len(training_args.validation))
                    extended_dataset = 'full' if uom_args.exp_forgetting_alphas is not None else 'flow_est'
                    for val_dataset in training_args.validation:
                        if val_dataset == 'kubric2024_val_subsplit' and is_validation_step(total_steps, debug=False):
                            progress.update(validation_task, description='[blue]Validation Kubric 2024')
                            results.update(validate_kubric_2024(model, flow_estimator, subsplit='validation', progress=progress,
                                                                uncertainty_loss_type=training_args.uncertainty_loss,
                                                                occlusion_loss_type=training_args.occlusion_loss,
                                                                extended_dataset=extended_dataset,
                                                                exp_forgetting_alphas=uom_args.exp_forgetting_alphas))
                            progress.update(validation_task, advance=1)
                        elif val_dataset == 'kubric2024_debug_subsplit':
                            progress.update(validation_task, description='[blue]Debug Kubric 2024')
                            results.update(validate_kubric_2024(model, flow_estimator, subsplit='debug', progress=progress,
                                                                uncertainty_loss_type=training_args.uncertainty_loss,
                                                                occlusion_loss_type=training_args.occlusion_loss,
                                                                extended_dataset=extended_dataset,
                                                                exp_forgetting_alphas=uom_args.exp_forgetting_alphas))
                            progress.update(validation_task, advance=1)
                        elif val_dataset == 'sintel_val_subsplit' and is_validation_step(total_steps, debug=False):
                            progress.update(validation_task, description='[blue]Validation Sintel')
                            results.update(validate_sintel(model, flow_estimator, subsplit='validation', progress=progress,
                                                                uncertainty_loss_type=training_args.uncertainty_loss,
                                                                occlusion_loss_type=training_args.occlusion_loss,
                                                                extended_dataset=extended_dataset,
                                                                exp_forgetting_alphas=uom_args.exp_forgetting_alphas))
                            progress.update(validation_task, advance=1)
                        elif val_dataset == 'kubric_val_subsplit' and is_validation_step(total_steps, debug=False):
                            progress.update(validation_task, description='[blue]Validation Kubric')
                            results.update(validate_kubric(model, flow_estimator, subsplit='validation', progress=progress,
                                                                uncertainty_loss_type=training_args.uncertainty_loss,
                                                                occlusion_loss_type=training_args.occlusion_loss,
                                                                extended_dataset=extended_dataset,
                                                                exp_forgetting_alphas=uom_args.exp_forgetting_alphas))
                            progress.update(validation_task, advance=1)

                    progress.update(validation_task, completed=True)
                    progress.remove_task(validation_task)
                    print('after val: ', total_steps, train_timer.iter(), train_timer())

                    logger.write_dict(results)

                    logger.write_images({'image1': image1, 'image2': image2, 'valid': valid_gt})
                    logger.write_images({'flow_gt': flow_gt})
                    if occl_gt is not None:
                        logger.write_images({'occl_gt': occl_gt})
                    logger.write_images({'flow_est': all_predictions['flow'][0]})
                    if 'occlusion' in all_predictions:
                        # OCCLUSION LOGGING
                        occl_predictions = all_predictions['occlusion']
                        occl_prediction_single = universal_prediction_fce(occl_predictions[-1],
                                                                          loss_type=training_args.occlusion_loss,
                                                                          threshold=None)
                        logger.write_images({'occl_est': occl_prediction_single})

                    if 'uncertainty' in all_predictions:
                        # UNCERTAINTY LOGGING
                        uncertainty_predictions = all_predictions['uncertainty']
                        sigma2 = torch.exp(uncertainty_predictions[-1])
                        logger.write_images({'sigma2_est': sigma2 * 255})
                        sigma2_minmax = (sigma2 - sigma2.min()) / (sigma2.max() - sigma2.min())
                        logger.write_images({'sigma2_est_minmax': sigma2_minmax * 255})
                        sigma = torch.sqrt(sigma2)
                        logger.write_images({'sigma_est': sigma * 255})
                        sigma_minmax = (sigma - sigma.min()) / (sigma.max() - sigma.min())
                        logger.write_images({'sigma_est_minmax': sigma_minmax * 255})

                    for unc_threshold in range(32):
                        unc_key = f'uncertainty{unc_threshold}'
                        if unc_key in all_predictions:
                            # OCCLUSION LOGGING
                            unc_predictions = all_predictions[unc_key]
                            unc_prediction_single = universal_prediction_fce(unc_predictions[-1],
                                                                             loss_type=training_args.uncertainty_loss,
                                                                             threshold=None)
                            logger.write_images({f'{unc_key}_est': 255. * unc_prediction_single})

                    # model = weight_freezer(model, args)
                    model.train()

                total_steps += 1
                progress.update(training_task, advance=1)

                if total_steps > training_args.num_steps:
                    should_keep_training = False
                    break

            logger.close()
            PATH = f'{args.checkpoints}/{args.name}.pth'
            torch.save(model.state_dict(), PATH)

            return PATH

    print('STOP')


if __name__ == '__main__':
    args = parse_arguments()
    train(args)
