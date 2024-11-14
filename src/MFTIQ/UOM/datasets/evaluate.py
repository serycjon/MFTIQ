import einops
import torch
import numpy as np
import MFTIQ.UOM.datasets.datasets as datasets
from MFTIQ.RAFT.core.utils.utils import InputPadder, forward_interpolate
from MFTIQ.UOM.losses.loss_and_prediction import universal_loss_fce, universal_prediction_fce


def uncertainty_loss(uncertainty_est, flow_est, flow_gt, uncertainty_loss_type=None):
    huber_loss = torch.nn.SmoothL1Loss(reduction='none')
    alpha = uncertainty_est
    exp_alpha = torch.exp(-alpha)
    loss = exp_alpha * huber_loss(flow_est, flow_gt) + 0.5 * alpha
    return loss.cpu()

def occlusion_loss(occl_est, occl_gt, occlusion_loss_type=None):
    occl_gt_thresholded = occl_gt > 0.5
    occl_gt_thresholded = einops.rearrange(occl_gt_thresholded[0:1,:,:], 'C H W -> 1 C H W')
    loss = universal_loss_fce(occl_est, occl_gt_thresholded, occlusion_loss_type)
    return torch.squeeze(loss).cpu()

def occlusion_accuracy(occl_est, occl_gt, occlusion_loss_type=None):
    occl_est = einops.rearrange(occl_est, 'C H W -> 1 C H W')
    occl_gt = einops.rearrange(occl_gt, 'C H W -> 1 C H W')
    pred = universal_prediction_fce(occl_est, loss_type=occlusion_loss_type)
    gt = occl_gt > 0.5
    accuracy = float((pred == gt).float().mean())
    return accuracy

def uncertainty_eval(uncertainty_est, flow_est, flow_gt, uncertainty_loss_type=None):
    gt_epe = einops.reduce(torch.square(flow_est.squeeze(dim=0) - flow_gt),
                           'xy H W -> 1 H W',
                           reduction='sum')
    pred_epe = uncertainty_est

    overshoot = (pred_epe > gt_epe).float().mean().cpu()
    diff = torch.abs(gt_epe - pred_epe)
    sub_1 = (diff < 1).float().mean().cpu()
    sub_5 = (diff < 5).float().mean().cpu()
    return overshoot, sub_1, sub_5

def uncertainty_ce_loss(uncertainty_est, flow_est, flow_gt, occl_gt, threshold, uncertainty_loss_type=None):
    flow_epe = torch.sqrt(torch.sum((flow_est - flow_gt) ** 2, dim=1, keepdim=False)).detach()
    flow_epe_mask = flow_epe >= threshold

    occl_gt_thresholded = occl_gt > 0.5
    occl_gt_thresholded = occl_gt_thresholded[0, :, :]

    unc_occl_mask = torch.logical_or(flow_epe_mask, occl_gt_thresholded)
    unc_occl_mask = einops.rearrange(unc_occl_mask, 'C H W -> 1 C H W')
    loss = universal_loss_fce(uncertainty_est, unc_occl_mask, loss_type=uncertainty_loss_type)
    return loss.cpu()

def uncertainty_ce_accuracy(uncertainty_est, flow_est, flow_gt, occl_gt, threshold, uncertainty_loss_type=None):
    # cross-entropy loss for (binary) uncertainty
    flow_epe = torch.sqrt(torch.sum((flow_est - flow_gt) ** 2, dim=1, keepdim=False)).detach()
    flow_epe_mask = flow_epe >= threshold

    occl_gt_thresholded = occl_gt > 0.5
    unc_occl_mask = torch.logical_or(flow_epe_mask, occl_gt_thresholded)
    unc_occl_mask = einops.rearrange(unc_occl_mask, 'C H W -> 1 C H W')
    pred = universal_prediction_fce(uncertainty_est, loss_type=uncertainty_loss_type)
    accuracy = float((pred == unc_occl_mask).float().mean())
    return accuracy

@torch.no_grad()
def validate_general(val_dataset, flow_estimator, uom_estimator, dataset_name, subset_name=None, n_val=20, quiet=False, progress=None,
                         occlusion_loss_type=None, uncertainty_loss_type=None):
    results = {}
    epe_list = []
    uncer_loss_list = []
    occl_loss_list = []
    occl_accuracy_list = []
    uncer_overshoot_list = []
    uncer_sub_1px_list = []
    uncer_sub_5px_list = []

    uncer_cross_entropy_dict_of_lists = {}
    uncer_accuracy_dict_of_lists = {}

    c_progress = None
    if progress is not None:
        c_progress = progress.add_task(f'[yellow]Validation {dataset_name} {subset_name if not None else ""}', total=n_val)

    for val_id in range(n_val):
        # get data
        c_data = val_dataset(val_id)
        image1, image2, flow_gt = c_data['img1'], c_data['img2'], c_data['flow']
        valid_gt, occl_gt = c_data['valid'], c_data['occl']

        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        flow_gt = flow_gt.cuda()
        valid_gt = valid_gt.cuda()
        occl_gt = occl_gt.cuda()

        # estimate optical flow
        padder = InputPadder(image1.shape)
        image1_flow, image2_flow = padder.pad(image1, image2)

        flow_est_dict = flow_estimator(image1_flow, image2_flow)
        flow_est = flow_est_dict['flow']
        flow_est = padder.unpad(flow_est)

        additional_data = {}
        if getattr(uom_estimator, 'exp_forgetting_alphas', None) is not None:
            for alpha in uom_estimator.exp_forgetting_alphas:
                key = f'flow_position_{int(100 * alpha):03d}'
                additional_data[key] = c_data[key]
                key = f'flow_speed_{int(100 * alpha):03d}'
                additional_data[key] = c_data[key]
                key = f'occl_{int(100 * alpha):03d}'
                additional_data[key] = c_data[key]

        for k, v in additional_data.items():
            additional_data[k] = v[None].cuda()

        # estimate UOM
        prediction_dict = uom_estimator(image1, image2, flow=flow_est, **additional_data)


        # for k in prediction_dict.keys():
        #     prediction_dict[k] = [prediction_dict[k]]

        # compute flow EPE
        flow = prediction_dict['flow']
        epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
        epe_list.append(epe.view(-1).cpu().numpy())

        if 'uncertainty' in prediction_dict:
            # compute uncertainty loss
            uncertainty = prediction_dict['uncertainty']
            uncer_loss = uncertainty_loss(uncertainty, flow, flow_gt, uncertainty_loss_type)
            uncer_loss_list.append(uncer_loss.view(-1).cpu().numpy())

            # compute uncertainty overshoot
            overshoot, sub_1, sub_5 = uncertainty_eval(uncertainty, flow, flow_gt, uncertainty_loss_type)
            uncer_overshoot_list.append(overshoot)
            uncer_sub_1px_list.append(sub_1)
            uncer_sub_5px_list.append(sub_5)

        if 'occlusion' in prediction_dict:
            occlusion = prediction_dict['occlusion']
            occl_loss = occlusion_loss(occlusion, occl_gt, occlusion_loss_type)
            occl_loss_list.append(occl_loss.view(-1).cpu().numpy())
            occl_accuracy_list.append(occlusion_accuracy(occlusion.squeeze(dim=0), occl_gt, occlusion_loss_type))

        for unc_threshold in range(32):
            if f'uncertainty{unc_threshold}' in prediction_dict:
                if not f'{unc_threshold}' in uncer_cross_entropy_dict_of_lists:
                    uncer_cross_entropy_dict_of_lists[f'{unc_threshold}'] = []
                    uncer_accuracy_dict_of_lists[f'{unc_threshold}'] = []

                uncertainty = prediction_dict[f'uncertainty{unc_threshold}']
                uncer_ce_loss = uncertainty_ce_loss(uncertainty, flow, flow_gt, occl_gt, unc_threshold, occlusion_loss_type)
                uncer_cross_entropy_dict_of_lists[f'{unc_threshold}'].append(uncer_ce_loss.view(-1).cpu().numpy())

                uncer_ce_accuracy = uncertainty_ce_accuracy(uncertainty, flow, flow_gt, occl_gt, unc_threshold, occlusion_loss_type)
                uncer_accuracy_dict_of_lists[f'{unc_threshold}'].append(uncer_ce_accuracy)
        if progress is not None:
            progress.update(c_progress, advance=1)
    if progress is not None:
        progress.update(c_progress, completed=True)
        progress.remove_task(c_progress)

    epe_all = np.concatenate(epe_list)
    epe = np.mean(epe_all)
    px1 = np.mean(epe_all < 1)
    px3 = np.mean(epe_all < 3)
    px5 = np.mean(epe_all < 5)
    if not quiet:
        print(f"Validation ({dataset_name}, {subset_name}) EPE: {epe}, 1px: {px1}, 3px: {px3}, 5px: {px5}")
    results[f'eval/{dataset_name}/flow {subset_name}'] = np.mean(epe_list)

    if len(uncer_loss_list) > 0:
        overshoot = np.mean(uncer_overshoot_list)
        sub_1 = np.mean(uncer_sub_1px_list)
        sub_5 = np.mean(uncer_sub_5px_list)
        results[f'eval/{dataset_name}/uncertainty overshoot {subset_name}'] = overshoot
        results[f'eval/{dataset_name}/uncertainty sub_1 {subset_name}'] = sub_1
        results[f'eval/{dataset_name}/uncertainty sub_5 {subset_name}'] = sub_5
        if not quiet:
            print(f"Validation ({dataset_name}, {subset_name}) EPE overshoot: {overshoot}, sub1: {sub_1}, sub5: {sub_5}")

    if len(occl_loss_list) > 0:
        occl_all = np.concatenate(occl_loss_list)
        occl_mean = np.mean(occl_all)
        results[f'eval/{dataset_name}/occl loss {subset_name}'] = occl_mean

        occl_mean = np.mean(occl_accuracy_list)
        results[f'eval/{dataset_name}/occl acc {subset_name}'] = occl_mean

        if not quiet:
            print(f"Validation ({dataset_name}, {subset_name}) OCCL_acc: {occl_mean}")

    for unc_threshold in range(32):
        if f'{unc_threshold}' in uncer_cross_entropy_dict_of_lists:
            if len(uncer_cross_entropy_dict_of_lists[f'{unc_threshold}']) > 0:
                unc_ce_all = np.concatenate(uncer_cross_entropy_dict_of_lists[f'{unc_threshold}'])
                unc_ce_mean = np.mean(unc_ce_all)
                results[f'eval/{dataset_name}/uncertainty-{unc_threshold} loss {subset_name}'] = unc_ce_mean

                occl_mean = np.mean(occl_accuracy_list)
                results[f'eval/{dataset_name}/occl acc {subset_name}'] = occl_mean

                unc_acc_all = np.array(uncer_accuracy_dict_of_lists[f'{unc_threshold}'])
                unc_acc_mean = np.mean(unc_acc_all)
                results[f'eval/{dataset_name}/uncertainty-{unc_threshold} acc {subset_name}'] = unc_acc_mean

                if not quiet:
                    print(f"Validation ({dataset_name}, {subset_name}) UNC_{unc_threshold}_acc: {unc_acc_mean}")

    return results


@torch.no_grad()
def validate_kubric_2024(model, flow_estimator, iters=12, n_val=None, subsplit=None, quiet=False, progress=None,
                         occlusion_loss_type=None, uncertainty_loss_type=None, **kwargs):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}

    val_dataset = datasets.KubricLong2024Dataset(split='train', subsplit=subsplit, load_occlusion=True, **kwargs)
    n_val = n_val if n_val is not None else len(val_dataset)
    val_lambda = lambda idx:val_dataset[idx]
    flow_estimator_lambda = lambda im1, im2: flow_estimator(im1, im2, iters=iters, test_mode=True)
    c_results = validate_general(val_lambda, flow_estimator_lambda, model, n_val=n_val,
                                 dataset_name='kubric2024', subset_name=subsplit, quiet=quiet, progress=progress,
                                 occlusion_loss_type=occlusion_loss_type, uncertainty_loss_type=uncertainty_loss_type)
    results.update(c_results)
    return results

@torch.no_grad()
def validate_sintel(model, flow_estimator, iters=12, n_val=None, subsplit=None, quiet=False, progress=None,
                         occlusion_loss_type=None, uncertainty_loss_type=None, **kwargs):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype, subsplit=subsplit, load_occlusion=True, **kwargs)
        n_val = n_val if n_val is not None else len(val_dataset)
        val_lambda = lambda idx:val_dataset[idx]
        flow_estimator_lambda = lambda im1, im2: flow_estimator(im1, im2, iters=iters, test_mode=True)
        c_results = validate_general(val_lambda, flow_estimator_lambda, model, n_val=n_val,
                                     dataset_name='sintel', subset_name=dstype, quiet=quiet, progress=progress,
                                 occlusion_loss_type=occlusion_loss_type, uncertainty_loss_type=uncertainty_loss_type)
        results.update(c_results)
    return results


@torch.no_grad()
def validate_kubric(model, flow_estimator, iters=12, n_val=20, subsplit=None, quiet=False, progress=None,
                         occlusion_loss_type=None, uncertainty_loss_type=None, **kwargs):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    val_dataset = datasets.KubricDataset(split=subsplit, load_occlusion=True, correct_flow=True, **kwargs)

    deltas = [1, 2, 4, 8, 16]
    for delta in deltas:
        n_val = n_val if n_val is not None else range(len(val_dataset))
        val_lambda = lambda idx: val_dataset.get_data_delta(idx, delta)
        flow_estimator_lambda = lambda im1, im2: flow_estimator(im1, im2, iters=iters, test_mode=True)
        c_results = validate_general(val_lambda, flow_estimator_lambda, model, n_val=n_val,
                                     dataset_name='kubric', subset_name=f'{delta}', quiet=quiet, progress=progress,
                                 occlusion_loss_type=occlusion_loss_type, uncertainty_loss_type=uncertainty_loss_type)
        results.update(c_results)
    return results
