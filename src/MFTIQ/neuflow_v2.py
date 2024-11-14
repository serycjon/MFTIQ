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
from MFTIQ.NeuFlow_v2.data_utils import frame_utils
from MFTIQ.NeuFlow_v2.NeuFlow.backbone_v7 import ConvBlock

def fuse_conv_and_bn(conv, bn):
        """Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/."""
        fusedconv = (
            torch.nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=conv.groups,
                bias=True,
            )
            .requires_grad_(False)
            .to(conv.weight.device)
        )

        # Prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

        # Prepare spatial bias
        b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        return fusedconv

class NeuFlowWrapper():
    @torch.inference_mode()
    def __init__(self, config):
        self.C = config

        self.padding_factor = 16

        self.device = torch.device('cuda')
        model = self.C.model()
        model.load_state_dict(torch.load(self.C.neuflow_weights, map_location='cpu')['model'],
                              strict=True)
        for m in model.modules():
            if type(m) is ConvBlock:
                m.conv1 = fuse_conv_and_bn(m.conv1, m.norm1)  # update conv
                m.conv2 = fuse_conv_and_bn(m.conv2, m.norm2)  # update conv
                delattr(m, "norm1")  # remove batchnorm
                delattr(m, "norm2")  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        model.to(self.device)

        model.cuda()
        model.eval()
        model.requires_grad_(False)
        model.half()

        self.model = model

    @iex
    @torch.inference_mode()
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
        image1, image2 = image1.cuda().half(), image2.cuda().half()

        padder = frame_utils.InputPadder(image1.shape, padding_factor=self.padding_factor)
        image1, image2 = padder.pad(image1, image2)

        self.model.init_bhwd(1, image1.shape[-2], image1.shape[-1], 'cuda')

        results = self.model(image1, image2)

        # useful when using parallel branches
        flow_pr = results[-1].float()

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
