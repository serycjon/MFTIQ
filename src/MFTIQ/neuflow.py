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
from MFTIQ.NeuFlow.data_utils import frame_utils


class NeuFlowWrapper():
    def __init__(self, config):
        self.C = config

        self.padding_factor = 16

        self.device = torch.device('cuda')
        model = self.C.model()
        model.load_state_dict(torch.load(self.C.neuflow_weights, map_location='cpu')['model'])
        model.to(self.device)

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

        H, W = src_img.shape[:2]
        # the_flow_timer = time_measurer('ms')
        image1 = einops.rearrange(torch.from_numpy(src_img[:, :, ::-1].copy()),
                                  'H W C -> 1 C H W', C=3)
        image2 = einops.rearrange(torch.from_numpy(dst_img[:, :, ::-1].copy()),
                                  'H W C -> 1 C H W', C=3)
        image1, image2 = image1.cuda().float(), image2.cuda().float()

        padder = frame_utils.InputPadder(image1.shape, padding_factor=self.padding_factor)
        image1, image2 = padder.pad(image1, image2)

        self.model.init_bhw(image1.shape[0], image1.shape[-2], image1.shape[-1])

        results = self.model(image1, image2)

        # useful when using parallel branches
        flow_pr = results[-1]

        flow_pre = padder.unpad(flow_pr)
        flow = flow_pre[0]  # xy, H, W
        assert flow.shape == (2, H, W)

        # Warp: [B,H,W,4] for all images in batch of size B, for each pixel HxW, we ouput the input and matching coordinate in the normalized grids [-1,1]x[-1,1]

        if mode == 'flow':
            if numpy_out:
                flow = ensure_numpy(flow)

            extra_outputs = {'occlusion': None,
                             'sigma': None}
            return flow, extra_outputs
        else:
            raise NotImplementedError("TC mode not implemented")
