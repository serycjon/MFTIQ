from __future__ import print_function, division

import torch
import torch.nn as nn
import einops

from MFTIQ.UOM.losses.loss_and_prediction import universal_loss_fce

# exclude extremly large displacements
MAX_FLOW = 1000

def optical_flow_loss(preds_dict, flow_gt, valid_gt, occl_gt=None,
                      preds_keys_dict=None,
                      flow_loss_type=None, occlusion_loss_type=None,
                      args=None, **kwargs):
    total_loss = 0.0
    metrics = {}

    flow_preds = preds_dict['flow']
    occl_preds = preds_dict['occlusion']
    prefix_flow_metric_name = preds_keys_dict['flow'] if preds_keys_dict is not None else None

    if not isinstance(flow_preds, list):
        flow_preds = [flow_preds]
    if not isinstance(occl_preds, list):
        occl_preds = [occl_preds]
    if prefix_flow_metric_name is not None and not isinstance(prefix_flow_metric_name, list):
        prefix_flow_metric_name = [prefix_flow_metric_name]

    return sequence_flow_loss(flow_preds, flow_gt, occl_preds, occl_gt, valid_gt,
                              flow_loss_type=flow_loss_type, occlusion_loss_type=occlusion_loss_type,
                              prefix_metric_name=prefix_flow_metric_name,
                              **kwargs)



def sequence_flow_loss(flow_preds, flow_gt, occl_preds, occl_gt, valid,
                       gamma=0.8, max_flow=MAX_FLOW, max_epe=MAX_FLOW,
                       flow_loss_type=None, occlusion_loss_type=None,
                       prefix_metric_name=None):
    """ Loss function defined over sequence of flow predictions """
    assert flow_gt.ndim == 4
    assert valid.ndim == 4
    assert occl_gt.ndim == 4

    # RAFT returned flowou predictions on each RAFT iteration, here we have only single prediction
    total_loss = 0.0
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
        flow_epe = einops.reduce((c_flow_preds - flow_gt) ** 2, 'B xy H W -> B 1 H W', reduction='sum',
                                 xy=2).sqrt().detach()

        flow_epe_valid_4occl = torch.logical_or(flow_epe <= max_epe, occl_gt > 0.5)
        # print(f"OCCL: fraction of valid pixels A {valid.float().mean():0.2f}")
        c_flow_valid = torch.logical_and(valid, occl_gt < 0.5)
        c_occl_valid = torch.logical_and(valid, flow_epe_valid_4occl)
        # print(f"OCCL: fraction of valid pixels B {valid.float().mean():0.2f}")

        # OCCL LOSS
        loss = universal_loss_fce(c_occl_preds, occl_gt, loss_type=occlusion_loss_type)
        assert valid.shape == loss.shape
        occl_loss = loss[c_occl_valid].mean()
        # total_occl_loss += occl_loss # <- occl_loss is not added to total loss
        total_loss += occl_loss # <- occl_loss is not added to total loss


        # OPTICAL FLOW LOSS
        loss = universal_loss_fce(c_flow_preds, flow_gt, loss_type=flow_loss_type)
        assert valid.shape == loss.shape
        flow_loss = loss[c_flow_valid].mean()
        total_loss += flow_loss

        # renaming metric for multiple inputs
        c_prefix_name = ''
        if prefix_metric_name is not None:
            c_prefix_name = f'{prefix_metric_name[idx]}/'
        metrics[f'train/{c_prefix_name}cross_entropy_occl'] = occl_loss.item()

        epe = torch.sum((flow_preds[idx] - flow_gt) ** 2, dim=1).sqrt()
        epe = epe.view(-1)[valid.view(-1)]

        metrics[f'train/{c_prefix_name}epe'] = epe.mean().item()
        metrics[f'train/{c_prefix_name}1px'] = (epe < 1).float().mean().item()
        metrics[f'train/{c_prefix_name}3px'] = (epe < 3).float().mean().item()
        metrics[f'train/{c_prefix_name}5px'] =(epe < 5).float().mean().item()

    return total_loss, metrics