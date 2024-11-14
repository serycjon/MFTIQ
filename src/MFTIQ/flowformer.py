"""A FlowFormer++ wrapper
mostly adopted from FlowFormerPlusPlus: visualize_flow.py
"""

import math
import torch
import torch.nn.functional as F
import einops
import numpy as np

from MFTIQ.FlowFormerPlusPlus.core.FlowFormer import build_flowformer
from MFTIQ.FlowFormerPlusPlus.core.utils.utils import InputPadder
from MFTIQ.utils.misc import ensure_numpy


class FlowFormerPPWrapper():
    def __init__(self, config):
        self.C = config

        model = torch.nn.DataParallel(build_flowformer(self.C.flowformer_cfg))
        model.load_state_dict(torch.load(self.C.flowformer_cfg.model))

        model.requires_grad_(False)
        model.cuda()
        model.eval()

        self.model = model

    def compute_flow(self, src_img, dst_img, mode='TC', vis=False, numpy_out=False, init_flow=None, **kwargs):
        """
        args:
            src_img: (H, W, 3) uint8 BGR opencv image
            dst_img: (H, W, 3) uint8 BGR opencv image
            mode: one of "flow" or "TC"
            init_flow: (xy H W) flow
        """
        assert kwargs.get('init_flow') is None

        weights = None
        image1 = torch.from_numpy(src_img[:, :, ::-1].copy()).permute(2, 0, 1).float()  # from H W BGR to RGB H W
        image2 = torch.from_numpy(dst_img[:, :, ::-1].copy()).permute(2, 0, 1).float()
        if init_flow is not None:
            if isinstance(init_flow, np.ndarray):
                init_flow = torch.from_numpy(init_flow.copy()).float()

        flow = _compute_flow(self.model, image1, image2, weights, init_flow=init_flow)  # (xy, H, W) cuda
        if mode == 'flow':
            if numpy_out:
                flow = ensure_numpy(flow)

            extra_outputs = {'occlusion': None,
                             'sigma': None,
                             'weights': None}
            return flow, extra_outputs
        else:
            raise NotImplementedError("TC mode not implemented")

TRAIN_SIZE = [432, 960]

def _compute_flow(model, image1, image2, weights=None, init_flow=None):
    image_size = image1.shape[1:]

    image1, image2 = image1[None].cuda(), image2[None].cuda()
    if init_flow is not None:
        init_flow = init_flow[None].cuda()

    hws = compute_grid_indices(image_size)
    if weights is None:     # no tile
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        if init_flow is not None:
            init_flow = padder.pad(init_flow)[0]
            init_flow = F.interpolate(init_flow, size=(init_flow.shape[2]//8, init_flow.shape[3]//8), mode='bilinear', align_corners=False) / 8.0

        flow_pre, _ = model(image1, image2, flow_init=init_flow)

        flow_pre = padder.unpad(flow_pre)
    else:                   # tile
        flows = 0
        flow_count = 0

        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]    
            if init_flow is not None:
                raise NotImplementedError('not ready for warm start')
            flow_pre, _ = model(image1_tile, image2_tile)
            padding = (w, image_size[1]-w-TRAIN_SIZE[1], h, image_size[0]-h-TRAIN_SIZE[0], 0, 0)
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)

        flow_pre = flows / flow_count
    flow = einops.rearrange(flow_pre, '1 xy H W -> xy H W', xy=2)
    return flow


def compute_grid_indices(image_shape, patch_size=TRAIN_SIZE, min_overlap=20):
    if min_overlap >= patch_size[0] or min_overlap >= patch_size[1]:
        raise ValueError(
            f"Overlap should be less than size of patch (got {min_overlap}"
            f"for patch size {patch_size}).")
    if image_shape[0] == patch_size[0]:
        hs = list(range(0, image_shape[0], patch_size[0]))
    else:
        hs = list(range(0, image_shape[0], patch_size[0] - min_overlap))
    if image_shape[1] == patch_size[1]:
        ws = list(range(0, image_shape[1], patch_size[1]))
    else:
        ws = list(range(0, image_shape[1], patch_size[1] - min_overlap))

    # Make sure the final patch is flush with the image boundary
    hs[-1] = image_shape[0] - patch_size[0]
    ws[-1] = image_shape[1] - patch_size[1]
    # hs[-1] = max(0, image_shape[0] - patch_size[0])
    # ws[-1] = max(0, image_shape[1] - patch_size[1])
    return [(h, w) for h in hs for w in ws]

def compute_weight(hws, image_shape, patch_size=TRAIN_SIZE, sigma=1.0, wtype='gaussian'):
    patch_num = len(hws)
    h, w = torch.meshgrid(torch.arange(patch_size[0]), torch.arange(patch_size[1]), indexing='ij')
    h, w = h / float(patch_size[0]), w / float(patch_size[1])
    c_h, c_w = 0.5, 0.5 
    h, w = h - c_h, w - c_w
    weights_hw = (h ** 2 + w ** 2) ** 0.5 / sigma
    denorm = 1 / (sigma * math.sqrt(2 * math.pi))
    weights_hw = denorm * torch.exp(-0.5 * (weights_hw) ** 2)

    weights = torch.zeros(1, patch_num, *image_shape)
    for idx, (h, w) in enumerate(hws):
        weights[:, idx, h:h+patch_size[0], w:w+patch_size[1]] = weights_hw
    weights = weights.cuda()
    patch_weights = []
    for idx, (h, w) in enumerate(hws):
        patch_weights.append(weights[:, idx:idx+1, h:h+patch_size[0], w:w+patch_size[1]])

    return patch_weights
