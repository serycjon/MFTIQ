import torch
import torch.nn as nn
from torchvision.ops.focal_loss import sigmoid_focal_loss as binary_focal_loss
import einops

def universal_loss_fce(est, gt, loss_type=None):
    assert est.ndim == 4
    assert gt.ndim == 4
    if loss_type is None or loss_type in ['cross_entropy', 'ce', 'unc_threshold']:
        cross_ent_loss = nn.CrossEntropyLoss(reduction='none')
        loss = cross_ent_loss(est, einops.rearrange(gt.long(), 'N 1 H W -> N H W'))
        loss = einops.rearrange(loss, 'N H W -> N 1 H W')
    elif loss_type in ['binary_cross_entropy', 'bce']:
        cross_ent_loss = nn.BCEWithLogitsLoss(reduction='none')
        loss = cross_ent_loss(est, gt.float())
    elif loss_type in ['binary_focal', 'focal_bce', 'focal_binary']:
        loss = binary_focal_loss(est, gt.float(), alpha=0.5, gamma=2, reduction='none')
    elif loss_type == 'L1':
        l1_loss = nn.L1Loss(reduction='none')
        loss = l1_loss(est, gt.float())
        loss = torch.sum(loss, dim=1, keepdim=True)
    else:
        raise NotImplementedError
    return loss

def universal_prediction_fce(est, loss_type=None, threshold=0.5):
    assert est.ndim == 4
    if loss_type is None or loss_type in ['cross_entropy', 'ce', 'unc_threshold']:
        prediction = est.softmax(dim=1)
        prediction =  prediction[:, 1:2, :, :]
    elif loss_type in ['binary_cross_entropy', 'bce', 'binary_focal', 'focal_bce', 'focal_binary']:
        prediction = est.sigmoid()
    elif loss_type == 'focal':
        prediction = est.softmax(dim=1)
    else:
        raise NotImplementedError

    if threshold is not None:
        return prediction > threshold
    else:
        return prediction

