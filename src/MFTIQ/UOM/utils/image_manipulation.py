# MFTIQ - WACV2025
import math
import itertools
from collections import OrderedDict
from functools import partial
import einops
import torch
import torch.nn.functional as F
import os

from scipy.interpolate import RegularGridInterpolator, griddata
from MFTIQ.utils import interpolation
from MFTIQ.utils.geom_utils import torch_get_featuremap_coords
from MFTIQ.utils.misc import ensure_torch, ensure_numpy
import numpy as np

if os.getenv('REMOTE_DEBUG'):
    import matplotlib
    matplotlib.use('module://backend_interagg')
    import matplotlib.pyplot as plt


class DownsampleFlow:
    def __init__(self, downsample_factor, align_corners=False):
        self.downsample_factor = downsample_factor
        self.align_corners = align_corners
        # assert downsample_factor in [1, 2, 4, 8, 16, 32]

    def __call__(self, flow):
        # Assuming 'flow' is a tensor of shape [batch_size, 2, height, width]
        # where flow[:, 0, :, :] is the horizontal component,
        # and flow[:, 1, :, :] is the vertical component.

        if self.downsample_factor == 1:
            return flow

        # Downsample the flow using average pooling
        downsampled_flow = F.avg_pool2d(flow, kernel_size=self.downsample_factor,
                                        stride=self.downsample_factor, count_include_pad=self.align_corners)

        # Scale the flow values to adjust to the new resolution
        downsampled_flow /= self.downsample_factor

        return downsampled_flow


class RotateAndPad(torch.nn.Module):
    def __init__(self, degrees):
        super().__init__()
        self.degrees = degrees
        self.orig_shape = None
        if self.degrees not in [0,90,180,270]:
            raise NotImplementedError(
                f'Rotation implemented only for 0, 90, 180 and 270 degrees, degrees={self.degrees} not implemented')

    def forward(self, img1, img2):
        self.orig_shape = einops.parse_shape(img1, 'H W C')
        if self.degrees == 0:
            return img1, img2
        elif self.degrees == 180:
            img2r = np.rot90(img2, 2)
            return img1, img2r
        elif self.degrees == 90:
            img2r = np.rot90(img2, 1)
        elif self.degrees == 270:
            img2r = np.rot90(img2, 3)
        else:
            raise NotImplementedError(f'Rotation implemented only for 0, 90, 180 and 270 degrees, degrees={self.degrees} not implemented')

        img1_shape = self.orig_shape
        img2r_shape = einops.parse_shape(img2r, 'H W C')
        _img1_x_pad = max(0, img2r_shape['W'] - img1_shape['W'])
        _img1_y_pad = max(0, img2r_shape['H'] - img1_shape['H'])
        _img2_x_pad = max(0, img1_shape['W'] - img2r_shape['W'])
        _img2_y_pad = max(0, img1_shape['H'] - img2r_shape['H'])

        img1_pad = np.pad(img1, ((0, _img1_y_pad), (0, _img1_x_pad), (0, 0)), mode='constant', constant_values=0)
        img2r_pad = np.pad(img2r, ((0, _img2_y_pad), (0, _img2_x_pad), (0, 0)), mode='constant', constant_values=0)

        return img1_pad, img2r_pad

    def correct_flow(self, flow, extra):
        if self.degrees == 0:
            return flow, extra

        coords1, coords2 = flow_to_coords(einops.rearrange(flow, 'C H W ->  1 H W C'))
        coords2_x, coords2_y = coords2[:, :, :, 0], coords2[:, :, :, 1]
        coords2_xc, coords2_yc, rot_key = None, None, None
        if self.degrees == 180:
            coords2_xc = (self.orig_shape['W'] - 1) - coords2_x
            coords2_yc = (self.orig_shape['H'] - 1) - coords2_y
            rot_key = 2
        elif self.degrees == 90:
            coords2_xc = (self.orig_shape['W'] - 1) - coords2_y
            coords2_yc = coords2_x
            rot_key = 3
        elif self.degrees == 270:
            coords2_xc = coords2_y
            coords2_yc = (self.orig_shape['H'] - 1) - coords2_x
            rot_key = 1
        else:
            NotImplementedError(f'Rotation implemented only for 0, 90, 180 and 270 degrees; not for {self.degrees}')

        coords2 = torch.stack([coords2_xc, coords2_yc], dim=-1)
        flow_corrected_bhwc = coords2 - coords1
        flow_corrected_chw = einops.rearrange(flow_corrected_bhwc, '1 H W C -> C H W')
        flow_corrected = flow_corrected_chw[:, :self.orig_shape['H'], :self.orig_shape['W']]
        # flowC_np = flow_corrected.cpu().detach().numpy()
        # flowCabs = np.abs(flowC_np)

        for key in extra.keys():
            # extra[key] = torch.rot90(extra[key], k=rot_key, dims=[1, 2])
            if extra[key] is None:
                continue
            extra[key] = extra[key][:, :self.orig_shape['H'], :self.orig_shape['W']]
        return flow_corrected, extra



class CenterPadding(torch.nn.Module):
    """
    copy from DINOv2
    added unpad
    https://github.com/facebookresearch/dinov2/blob/main/notebooks/semantic_segmentation.ipynb
    """
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    def forward(self, x):
        pad = self.init_padding(x)
        output = F.pad(x, pad)
        return output

    def init_padding(self, x):
        self._pad = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        return self._pad

    def get_left_pad(self):
        return self._pad[0]

    def get_top_pad(self):
        return self._pad[2]

    def unpad(self, *args):
        outputs = []
        for x in args:
            ht, wd = x.shape[-2:]
            c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
            outputs.append(x[..., c[0]:c[1], c[2]:c[3]])
        if len(outputs) == 1:
            return outputs[0]
        return outputs


def warp_backward(x, flow):
    """ Backward warp of features/image x according to flow
        Args:
            x: data [B, C, H, W]; torch.float32
            flow: flow [B, C, H, W] or [B, H, W, C], C = 2; torch.float32
        Returns: warped data [B, C, H, W]; torch.float32
        """
    assert x.ndim == 4
    assert flow.ndim == 4
    assert flow.shape[3] == 2 or flow.shape[1] == 2
    if flow.shape[1] == 2:
        flow = einops.rearrange(flow, 'B C H W -> B H W C')
    assert x.shape[2:] == flow.shape[1:3]

    H, W = x.shape[2:]
    device = flow.device
    flow_shape = einops.parse_shape(flow, 'B H W xy')


    grid_left_x = torch_get_featuremap_coords((flow_shape['H'], flow_shape['W']), device=device, keep_shape=True)
    grid_left_x = einops.rearrange(grid_left_x, 'xy H W -> H W xy', xy=2)
    grid_right_x = einops.repeat(grid_left_x, 'H W xy -> B H W xy', xy=2, B=flow_shape['B']) + flow
    grid_right_x = interpolation.normalize_coords(grid_right_x, H=H, W=W)
    warped_x = F.grid_sample(x.to(grid_right_x.dtype), grid_right_x, align_corners=True)
    return warped_x


def flow_to_coords(flow):
    device = flow.device
    flow_shape = einops.parse_shape(flow, 'B H W xy')
    coord1 = torch_get_featuremap_coords((flow_shape['H'], flow_shape['W']), device=device, keep_shape=True)
    coord1 = einops.rearrange(coord1, 'xy H W -> H W xy', xy=2)
    coord1 = einops.repeat(coord1, 'H W xy -> B H W xy', xy=2, B=flow_shape['B']).to(flow)
    coord2 = coord1 + flow
    return coord1, coord2

def sample_with_coords(x, coords, align_corners=False, orig_shape=None):
    if orig_shape is not None:
        H, W = orig_shape['H'], orig_shape['W']
    else:
        H, W = coords.shape[1:3]

    coords = interpolation.normalize_coords(coords, H=H, W=W, align_corners=align_corners)
    return F.grid_sample(x.to(coords.dtype), coords, align_corners=align_corners)

def universal_warp_backward(x1, x2,
                            flow=None, coords1=None, coords2=None,
                            x1_orig_shape=None, x2_orig_shape=None,
                            x1_padding_up=None, x1_padding_left=None, x2_padding_up=None, x2_padding_left=None,
                            align_corners=False,
                            flow_downsample_coef=1, cost_volume_displacement=None):
    """ Backward warp of features/image x according to flow
        Args:
            x1: data - reference frame [B, C, H, W]; torch.float32
            x2: data - target frame [B, C, H, W]; torch.float32
            flow: flow [B, C, H, W] or [B, H, W, C], C = 2; torch.float32
            coords1: grid, if flow is not None - coords1 is ignored [B, C, H, W] or [B, H, W, C], C = 2; torch.float32
            coords2: grid + flow, if flow is not None - coords2 is ignored, [B, C, H, W] or [B, H, W, C], C = 2; torch.float32
            x1_orig_shape: parsed shape of x1 in original resolution; dict; optional
            x2_orig_shape: parsed shape of x2 in original resolution; dict; optional
            padding_up: if set add padding to the y-coordinates
            padding_left: if set add padding to the x-coordinates
            align_corners: (bool) use True  if the center of the top left pixel has coords [0, 0],
                                  use False if the center of the top left pixel has coords [0.5, 0.5],
            flow_downsample_coef: (int) downsampling coefficient for optical flow (1 = input resolution)
            cost_volume_displacement: (int, None) if set, create third output with warped x2 with augmented flow -
                                      cartesian product of displacements [-cost_volume_displacement,
                                      cost_volume_displacement] x [-cost_volume_displacement, cost_volume_displacement]
        Returns: warped data [B, C, H, W]; torch.float32
        """
    assert x1.ndim == 4 and x2.ndim == 4
    coords1, coords2 = flow_to_coords_with_padding(flow=flow, coords1=coords1, coords2=coords2,
                                                   x1_padding_up=x1_padding_up, x1_padding_left=x1_padding_left,
                                                   x2_padding_up=x2_padding_up, x2_padding_left=x2_padding_left,
                                                   align_corners=align_corners)

    if flow_downsample_coef is not None and flow_downsample_coef != 1:
        raise NotImplementedError

    warped_x1 = sample_with_coords(x1, coords1, align_corners=align_corners, orig_shape=x1_orig_shape)
    warped_x2 = sample_with_coords(x2, coords2, align_corners=align_corners, orig_shape=x2_orig_shape)

    if cost_volume_displacement:
        with torch.no_grad():
            cost_volume = []
            for d_left in range(-cost_volume_displacement, cost_volume_displacement+1):
                for d_up in range(-cost_volume_displacement, cost_volume_displacement+1):
                    if d_left == 0 and d_up == 0:
                        continue # skip default displacement
                    d_coords2 = create_coords_displacement(coords2, d_left, d_up)
                    d_warped_x2 = sample_with_coords(x2.detach(), d_coords2, align_corners=align_corners)
                    cost_volume.append(dot_product(warped_x1.detach(), d_warped_x2).detach())
            cost_volume = torch.concat(cost_volume, dim=1).detach()
        return warped_x1, warped_x2, cost_volume
    return warped_x1, warped_x2, None


def create_coords_displacement(coords, left, up):
    displacement = torch.ones_like(coords).to(coords.device)
    d_left, d_up = displacement[:, :, :, 0] * left, displacement[:, :, :, 1] * up
    displacement = torch.stack([d_left, d_up], dim=3)
    return coords + displacement


def compensate_padding_for_coords(coords, padding_up, padding_left):
    padding_up = 0 if padding_up is None else padding_up
    padding_left = 0 if padding_left is None else padding_left
    padding = torch.ones_like(coords).to(coords.device)
    p_left, p_up = padding[:, :, :, 0] * padding_left, padding[:, :, :, 1] * padding_up
    padding = torch.stack([p_left, p_up], dim=3)
    coords = coords + padding
    return coords

def flow_to_coords_with_padding(flow=None, coords1=None, coords2=None,
                                x1_padding_up=None, x1_padding_left=None, x2_padding_up=None, x2_padding_left=None,
                                align_corners=False):
    assert flow.ndim == 4
    assert flow is not None or (coords1 is not None and coords2 is not None)
    cc_list = []
    for cc in [flow, coords1, coords2]:
        if cc is not None:
            assert cc.shape[3] == 2 or cc.shape[1] == 2
            if cc.shape[1] == 2:
                cc = einops.rearrange(cc, 'B C H W -> B H W C')
        cc_list.append(cc)
    flow, coords1, coords2 = cc_list

    if flow is not None:
        coords1, coords2 = flow_to_coords(flow)

    if x1_padding_up is not None or x1_padding_left is not None:
        coords1 = compensate_padding_for_coords(coords1, x1_padding_up, x1_padding_left)

    if x2_padding_up is not None or x2_padding_left is not None:
        coords2 = compensate_padding_for_coords(coords2, x2_padding_up, x2_padding_left)

    if not align_corners:
        coords1 = coords1 + 0.5
        coords2 = coords2 + 0.5

    return coords1, coords2

def normalise_coords(coords, align_corners=False, orig_shape=None):
    if orig_shape is not None:
        H, W = orig_shape['H'], orig_shape['W']
    else:
        H, W = coords.shape[1:3]
    return interpolation.normalize_coords(coords, H=H, W=W, align_corners=align_corners)

def normalise_flow_and_coords(flow=None, coords1=None, coords2=None,
                              x1_orig_shape=None, x2_orig_shape=None,
                              x1_padding_up=None, x1_padding_left=None, x2_padding_up=None, x2_padding_left=None,
                              align_corners=False, keep_shape=False):

    coords1, coords2 = flow_to_coords_with_padding(flow=flow, coords1=coords1, coords2=coords2,
                                                   x1_padding_up=x1_padding_up, x1_padding_left=x1_padding_left,
                                                   x2_padding_up=x2_padding_up, x2_padding_left=x2_padding_left,
                                                   align_corners=align_corners)

    H, W = coords1.shape[1:3]
    norm_coords1 = normalise_coords(coords1, align_corners=align_corners, orig_shape=x1_orig_shape)
    norm_coords2 = normalise_coords(coords2, align_corners=align_corners, orig_shape=x2_orig_shape)

    if keep_shape:
        norm_coords1 = einops.rearrange(norm_coords1, 'B H W C -> B C H W')
        norm_coords2 = einops.rearrange(norm_coords2, 'B H W C -> B C H W')

    norm_flow = norm_coords2 - norm_coords1

    return norm_flow, norm_coords1, norm_coords2


def dot_product(x1, x2):
    es = torch.einsum('B C H W,B C H W -> B H W', x1, x2)
    return einops.rearrange(es, 'B H W -> B 1 H W')


def compute_diff(x1, x2, abs=False):
    diff = x1 - x2
    if abs:
        diff = torch.abs(diff)
    return einops.reduce(diff, 'B C H W -> B 1 H W', reduction='sum')