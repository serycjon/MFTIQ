from __future__ import print_function, division
# MFTIQ - WACV2025

import torch
import einops

from MFTIQ.UOM.losses.loss_and_prediction import universal_loss_fce

# exclude extremly large displacements
MAX_FLOW = 1000

def unc_thr_loss(preds_dict, flow_gt, valid, occl_gt=None, gamma=0.8, max_flow=MAX_FLOW,
                 occlusion_loss_type=None, uncertainty_loss_type=None,
                 preds_keys_dict=None,
                 args=None, **kwargs):
    total_loss = 0.0
    metrics = {}

    flow_preds = preds_dict['flow']
    occl_preds = preds_dict['occlusion']
    prefix_occl_metric_name = preds_keys_dict['occlusion'] if preds_keys_dict is not None else None

    unc_threshold_max = 0
    threshold_counter = 0
    for unc_threshold in range(30):
        key = f'uncertainty{unc_threshold}'
        if key not in preds_dict:
            continue
        prefix_unc_metric_name = preds_keys_dict[key] if preds_keys_dict is not None else None

        uncertainty_preds = preds_dict[key]
        uncertainty_loss, uncertainty_metrics = sequence_threshold_uncertainty_loss(flow_preds, uncertainty_preds,
                                                                      flow_gt, valid, gamma=gamma,
                                                                      max_flow=max_flow,
                                                                      threshold=unc_threshold,
                                                                      occl_gt=occl_gt,
                                                                      uncertainty_loss_type=uncertainty_loss_type,
                                                                      prefix_metric_name=prefix_unc_metric_name)
        metrics.update(uncertainty_metrics)
        total_loss += uncertainty_loss
        unc_threshold_max = unc_threshold
        threshold_counter += 1
    assert unc_threshold_max > 0, """Uncertainty Threshold Loss cannot be satisfied. No outputs to work with."""

    total_loss = total_loss / threshold_counter

    occl_loss, occl_metrics = sequence_occl_loss(occl_preds, occl_gt, flow_preds, flow_gt,
                                                 valid, gamma=gamma, max_flow=max_flow,
                                                 max_epe=unc_threshold_max,
                                                 occlusion_loss_type=occlusion_loss_type,
                                                 prefix_metric_name=prefix_occl_metric_name)
    metrics.update(occl_metrics)
    total_loss += occl_loss

    return total_loss, metrics

def sequence_occl_loss(occl_preds, occl_gt, flow_preds,
                       flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW, max_epe=MAX_FLOW,
                       occlusion_loss_type=None, prefix_metric_name=None):
    """ Loss function defined over sequence of flow predictions """
    assert flow_gt.ndim == 4
    assert valid.ndim == 4
    assert occl_gt.ndim == 4

    # RAFT returned flowou predictions on each RAFT iteration, here we have only single prediction
    total_occl_loss = 0.0
    metrics = {}

    # exlude invalid pixels and extremely large diplacements
    mag = einops.reduce(flow_gt ** 2, 'B xy H W -> B 1 H W', reduction='sum', xy=2).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)
    # only 100% occluded and 100% non-occluded are used for training
    occl_valid = torch.logical_or(occl_gt < 0.01, occl_gt > 0.99)
    valid = torch.logical_and(occl_valid, valid)

    occl_gt = occl_gt > 0.5

    for idx in range(len(occl_preds)):
        c_flow_preds = flow_preds[idx]
        c_occl_preds = occl_preds[idx]
        assert c_flow_preds.ndim == 4
        assert c_occl_preds.ndim == 4

        # Valid If FLOW_EPE < MAX_EPE or Occluded
        flow_epe = einops.reduce((c_flow_preds - flow_gt)**2, 'B xy H W -> B 1 H W', reduction='sum', xy=2).sqrt().detach()

        flow_epe_valid = torch.logical_or(flow_epe <= max_epe, occl_gt > 0.5)
        # print(f"OCCL: fraction of valid pixels A {valid.float().mean():0.2f}")
        c_valid = torch.logical_and(valid, flow_epe_valid)
        # print(f"OCCL: fraction of valid pixels B {valid.float().mean():0.2f}")

        loss = universal_loss_fce(c_occl_preds, occl_gt, loss_type=occlusion_loss_type)
        assert c_valid.shape == loss.shape
        occl_loss = loss[c_valid].mean()
        total_occl_loss += occl_loss

        # renaming metric for multiple occl inputs
        c_prefix_name = ''
        if prefix_metric_name is not None:
            c_prefix_name = f'{prefix_metric_name[idx]}/'
        metric_name = f'train/{c_prefix_name}cross_entropy_occl'
        metrics[metric_name] = occl_loss.item()

    return total_occl_loss, metrics

def sequence_threshold_uncertainty_loss(flow_preds, uncertainty_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW,
                              threshold=1.0, occl_gt=None, uncertainty_loss_type=None, prefix_metric_name=None, **kwargs):
    """ Loss function for thresholded uncertainty """
    assert flow_gt.ndim == 4
    assert valid.ndim == 4
    assert occl_gt is None or occl_gt.ndim == 4

    total_unc_loss = 0.0
    metrics = {}

    # exlude invalid pixels and extremely large diplacements
    mag = einops.reduce(flow_gt ** 2, 'B xy H W -> B 1 H W', reduction='sum', xy=2).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)
    # only 100% occluded and 100% non-occluded are used for training
    occl_valid = torch.logical_or(occl_gt < 0.01, occl_gt > 0.99)
    valid = torch.logical_and(occl_valid, valid)
    occl_gt = occl_gt > 0.5

    # RAFT returned flowou predictions on each RAFT iteration, here we have only single prediction
    for idx in range(len(uncertainty_preds)):
        c_flow_preds = flow_preds[idx]
        c_uncertainty_preds = uncertainty_preds[idx]
        assert c_flow_preds.ndim == 4
        assert c_uncertainty_preds.ndim == 4

        flow_epe = einops.reduce((c_flow_preds - flow_gt)**2, 'B xy H W -> B 1 H W', reduction='sum', xy=2).sqrt().detach()
        flow_epe_high_mask = flow_epe >= threshold
        no_match_mask = torch.logical_or(flow_epe_high_mask, occl_gt)

        i_loss = universal_loss_fce(c_uncertainty_preds, no_match_mask, loss_type=uncertainty_loss_type)
        assert i_loss.shape == valid.shape
        unc_loss = i_loss[valid].mean()
        total_unc_loss += unc_loss

        # renaming metric for multiple occl inputs
        c_prefix_name = ''
        if prefix_metric_name is not None:
            c_prefix_name = f'{prefix_metric_name[idx]}/'
        metric_name = f'train/{c_prefix_name}uncert_{threshold}'
        metrics[metric_name] = unc_loss.item()

    return total_unc_loss, metrics
