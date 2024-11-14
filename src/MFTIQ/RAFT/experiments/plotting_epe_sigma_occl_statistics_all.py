from __future__ import print_function, division
import sys, os
from RAFT.core.raft import RAFT
import tqdm

# sys.path.append('core')

import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from RAFT.core.datasets import MpiSintel, KubricDataset
from RAFT.core.utils import flow_viz
from RAFT.core.utils.utils import InputPadder
from RAFT.core.utils.flow_viz import flow_to_color
import einops

from RAFT.core.utils.statistics import CombinedStatistics

import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def mkdir_if_not_exist(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def mkdir_from_full_file_path_if_not_exist(path):
    basename = os.path.basename(path)
    mkdir_if_not_exist(path[:-len(basename)])

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__.update(kwargs)
def main(args):
    # Load RAFT
    device = f'cuda:{args.gpus}'

    raft_kwargs = {
        'occlusion_module': 'separate_with_uncertainty' if args.occlusion_module is None else args.occlusion_module,
        'restore_ckpt': args.model,
        'small': False,
        'mixed_precision': False,
        'alternate_corr': True,
    }
    raft_params = AttrDict(**raft_kwargs)

    # Of course, you can use argparse instead of attr_dict
    raft = torch.nn.DataParallel(RAFT(raft_params))
    raft.load_state_dict(torch.load(args.model, map_location='cpu'))

    raft = raft.module
    raft.requires_grad_(False)
    raft.to(device)
    raft.eval()

    # DATASET WITH GT FLOW and OCCL
    kubric_validation = KubricDataset(split='validation', load_occlusion=True, correct_flow=True)
    sintel_clean = MpiSintel(split='training', dstype='clean', load_occlusion=True, subsplit=args.subsplit)
    sintel_final = MpiSintel(split='training', dstype='final', load_occlusion=True, subsplit=args.subsplit)

    subsplit_name = '' if args.subsplit is None else f'_{args.subsplit}'
    datasets = {'kubric_val_corrected_flow': kubric_validation,
                f'sintel_clean{subsplit_name}': sintel_clean,
                f'sintel_final{subsplit_name}': sintel_final}

    for dataset_name, data in datasets.items():
        print(f'Processing dataset {dataset_name}:')

        mkdir_if_not_exist(args.save_path)

        delta_max = 23 if 'kubric' in dataset_name else 1
        comboStats = CombinedStatistics(dataset_name, save_root=args.save_path,
                                        model_name=os.path.basename(args.model)[:-4],
                                        ndelta=delta_max)

        if 'kubric' in dataset_name:
            c_data = data.multi_image_list
            c_len = 50
        else:
            c_data = data
            c_len = len(c_data)

        for idx in tqdm.trange(c_len):
            if 'kubric' in dataset_name:
                im1o0, im2o0, flow_gt0, valid0, occl_gt0 = data.get_data_delta(idx, 0)
                obj_gt_template = flow_gt0[0:1, :, :] > 0.2
            else:
                obj_gt_template = None

            for delta in range(1,delta_max+1):
                if 'kubric' in dataset_name:
                    im1o, im2o, flow_gt, valid, occl_gt = data.get_data_delta(idx, delta)
                else:
                    im1o, im2o, flow_gt, valid, occl_gt = data[idx]
                c_im1_path = data.get_reference_frame_path(idx, relative=True)

                im1 = torch.unsqueeze(im1o, 0)
                im2 = torch.unsqueeze(im2o, 0)

                padder = InputPadder(im1.shape, mode='sintel')
                im1, im2 = padder.pad(im1, im2)

                all_predictions = raft(im1.to(device), im2.to(device), iters=args.iters, test_mode=True)
                flow_pred = torch.squeeze(padder.unpad(all_predictions['flow']), dim=0)
                occl_pred = torch.squeeze(padder.unpad(all_predictions['occlusion'].softmax(dim=1)[:,1:2,:,:]), dim=0)
                uncertainty_pred = torch.squeeze(padder.unpad(all_predictions['uncertainty']), dim=0)

                comboStats.add_data(flow_pred, flow_gt, valid, occl_pred, occl_gt, uncertainty_pred, delta=delta, fg_mask=obj_gt_template)

            if args.debug:
                if idx >= 10:
                    break

        comboStats.save_graphs()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/home.stud/neoramic/repos/raft_new_debug/RAFT/checkpoints/50000_raft-things-sintel-occlusion-uncertainty-base-sintel.pth', help="restore checkpoint")
    parser.add_argument('--save_path', default='/datagrid/personal/neoral/RAFT_occl_uncertainty_outputs', help="saving_path")
    parser.add_argument('--iters', type=int, default=24, help='number of iterations')
    parser.add_argument('--gpus', type=int, default=0, help='gpu number')
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--occlusion_module', type=str, default=None,
                        choices=[None, 'separate', 'with_uncertainty', 'separate_with_uncertainty',
                                 'separate_with_uncertainty_upsample8',
                                 'separate_with_uncertainty_upsample8_morelayers'])
    parser.add_argument('--subsplit', type=str, default=None, choices=[None, 'training', 'validation'])

    args = parser.parse_args()

    main(args)