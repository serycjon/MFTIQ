"""A DKM wrapper
mostly adopted from DKM: demo_match.py
"""

import math
import torch
import torch.nn.functional as F
import einops
from PIL import Image
from ipdb import iex

from MFTIQ.utils.misc import ensure_numpy


class DKMWrapper():
    def __init__(self, config):
        self.C = config

        self.device = torch.device('cuda')
        model = self.C.model(device=self.device)

        model.requires_grad_(False)
        model.cuda()
        model.eval()

        self.model = model

    @iex
    def compute_flow(self, src_img, dst_img, mode='TC', vis=False, numpy_out=False, **kwargs):
        """
        args:
            src_img: (H, W, 3) uint8 BGR opencv image
            dst_img: (H, W, 3) uint8 BGR opencv image
            mode: one of "flow" or "TC"
            init_flow: (xy H W) flow
        """
        assert kwargs.get('init_flow') is None

        image1 = Image.fromarray(src_img[:, :, ::-1].copy())  # from H W BGR to RGB H W
        image2 = Image.fromarray(dst_img[:, :, ::-1].copy())

        orig_H, orig_W = src_img.shape[:2]
        H, W = 864, 1152 # constants used in DKM

        image1 = image1.resize((W, H))
        image2 = image2.resize((W, H))

        self.model.symmetric = False
        warp, certainty = self.model.match(image1, image2, device=self.device)
        # print(f"warp.shape: {warp.shape}")
        # symmetric => (864, 2 * 1152, 4)
        # non-symmetric => (864, 1152, 4)
        # print(f"certainty.shape: {certainty.shape}")
        # symmetric => (864, 2 * 1152)
        # non-symmetric => (1, 864, 1152)

        raster_coords = to_raster_coords(warp, H, W)
        flow = raster_to_flow(raster_coords, H, W, orig_H, orig_W)
        if self.model.symmetric:
            certainty = einops.rearrange(certainty, 'H W -> 1 H W')
            
        certainty = einops.rearrange(
            F.interpolate(einops.rearrange(certainty, '1 H W -> 1 1 H W'),
                          size=(orig_H, orig_W), mode='bilinear', align_corners=False),
            '1 1 H W -> 1 H W')

        # Warp: [B,H,W,4] for all images in batch of size B, for each pixel HxW, we ouput the input and matching coordinate in the normalized grids [-1,1]x[-1,1]
        # Certainty: [B,H,W] a number in each pixel indicating the matchability of the pixel.

        if mode == 'flow':
            if numpy_out:
                flow = ensure_numpy(flow)
                certainty = ensure_numpy(certainty)

            extra_outputs = {'occlusion': 1 - certainty,
                             'sigma': 1 - certainty}
            return flow, extra_outputs
        else:
            raise NotImplementedError("TC mode not implemented")

def to_raster_coords(warp, H, W):
    """Convert to raster coordinates (top-left = 0.5, 0.5)"""
    result = warp.clone()
    result[:, :, 0] = (W / 2) * (warp[:, :, 0] + 1)
    result[:, :, 1] = (H / 2) * (warp[:, :, 1] + 1)
    result[:, :, 2] = (W / 2) * (warp[:, :, 2] + 1)
    result[:, :, 3] = (H / 2) * (warp[:, :, 3] + 1)

    return result

def raster_to_flow(raster_warp, H, W, H_orig, W_orig):
    # first two coordinates are raster coordinate grid (starting from 0.5, 0.5)
    # we can skip these and only extract the last two coordinates
    raster_flow = raster_warp[:, :, 2:] - raster_warp[:, :, :2]

    # scale the flow back to original resolution
    H_scale = H_orig / H
    W_scale = W_orig / W

    raster_flow[:, :, 0] *= W_scale
    raster_flow[:, :, 1] *= H_scale

    # now we convert the flow from raster to pixel grid coordinates
    # pixel_flow = raster_flow - 0.5

    # ^ this should be the correct thing to do, but when applied in
    # chain-only MFT, the results slowly drift to topleft... without
    # the -0.5, the results seem to be stable
    pixel_flow = raster_flow

    # scale the flow field back to originial resolution
    pixel_flow_up = F.interpolate(einops.rearrange(pixel_flow, 'H W xy -> 1 xy H W', xy=2),
                                  size=(H_orig, W_orig), mode='bilinear', align_corners=False)
    
    return einops.rearrange(pixel_flow_up, '1 xy H W -> xy H W', xy=2)
