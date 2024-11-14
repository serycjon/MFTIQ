"""A RoMa wrapper
mostly adopted from DKM: demo_match.py
"""
import warnings
import torch
import torch.nn.functional as F
import einops
from PIL import Image
from ipdb import iex

from MFTIQ.utils.misc import ensure_numpy
from MFTIQ.dkm import to_raster_coords, raster_to_flow

class RoMaWrapper():
    def __init__(self, config):
        self.C = config

        self.device = torch.device('cuda')
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore',
                                    category=UserWarning,
                                    message=r'Arguments other than a weight enum.*')
            warnings.filterwarnings(action='ignore',
                                    category=UserWarning,
                                    message=r"The parameter 'pretrained' is deprecated since 0.13.*")
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
        H, W = self.model.get_output_resolution()

        image1 = image1.resize((W, H))
        image2 = image2.resize((W, H))

        self.model.symmetric = False
        warp, certainty = self.model.match(image1, image2, device=self.device)
        # print(f"warp.shape: {warp.shape}")
        # print(f"certainty.shape: {certainty.shape}")

        raster_coords = to_raster_coords(warp, H, W)
        flow = raster_to_flow(raster_coords, H, W, orig_H, orig_W)
            
        certainty = einops.rearrange(
            F.interpolate(einops.rearrange(certainty, 'H W -> 1 1 H W'),
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
