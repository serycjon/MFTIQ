from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
import einops

from scipy.interpolate import RegularGridInterpolator
from MFTIQ.utils.geom_utils import torch_get_featuremap_coords


@lru_cache(maxsize=5)
def _get_scales(H, W, align_corners, device):
    """cached code used in normalize_coords"""
    if align_corners:
        scales = np.array([2 / (W - 1), 2 / (H - 1)]).astype(np.float32)  # maps to [0, 2]
    else:
        scales = np.array([2 / W, 2 / H]).astype(np.float32)  # maps to [0, 2]
    scales = torch.from_numpy(scales).to(device)
    scales = einops.rearrange(scales, '(N H W xy) -> N H W xy', N=1, H=1, W=1)
    return scales

def normalize_coords(coordinates, H, W, align_corners=True):
    """Normalize coordinates to be in [-1, 1] range as usedi in F.grid_sample.
    args:
        coordinates: (N H W xy) coordinates
        align_corners: (bool) use True if the center of the top left pixel has coords [0, 0],
                              use False if the center of the top left pixel has coords [0.5, 0.5],
    """
    device = coordinates.device
    scales = _get_scales(H, W, align_corners, device)
    return coordinates * scales - 1      # maps to [-1, 1]


def bilinear_sample(data, coords, device=None):
    """
    args:
        data: (batch, C, H, W) tensor
        coords: (batch, ...outshape..., xy) tensor of coordinates
    returns:
        sampled: (batch, ...outshape..., C) tensor
    """
    assert len(coords.shape) >= 3
    assert coords.shape[-1] == 2
    if device is None:
        device = coords.device
    data_shape = einops.parse_shape(data, 'batch C H W')
    norm_coords = normalize_coords(coords.to(device), data_shape['H'], data_shape['W'])
    flat_coords = norm_coords.view(coords.shape[0], -1, coords.shape[-1])  # flatten all the outshape
    grid_coords = einops.rearrange(flat_coords, 'batch N xy -> batch 1 N xy', xy=2)  # simulate HxW coord grid (but only 1xN)
    sampled_flat = F.grid_sample(data.to(device), grid_coords, align_corners=True)
    sampled = sampled_flat.view(*coords.shape[:-1], data_shape['C'])
    return sampled



class FlowInterpolator(object):
    def __init__(self, flow, additional_data=None):
        """
            flow: (H, W, 2) array of dx, dy pairs
            additional_data: (H, W, [C]) array.
        """
        H, W, C = flow.shape
        assert C == 2
        flow_grid_ys = np.arange(H)
        flow_grid_xs = np.arange(W)
        if additional_data is None:
            data = flow
        else:
            if len(additional_data.shape) < 3:
                additional_data = additional_data[:, :, np.newaxis]
            data = np.concatenate((flow, additional_data), axis=2)
        self.interp = RegularGridInterpolator((flow_grid_ys, flow_grid_xs), data, bounds_error=False, fill_value=np.nan)

    def __call__(self, positions, method='linear'):
        """
        args:
                positions: (N, 2) array of x, y pairs (possibly non-integer)
        """
        return self.interp(positions[:, ::-1], method=method)  # the scipy interpolator wants y, x coordinates


def bilinear_splat(data, data_coords, grid_shape):
    """
    Bilinear splat the data (at data_coords) onto a grid

    args:
        data: (N, C) tensor with data to be splatted
        data_coords: (N, 2) tensor with xy coordinates of the data
        grid_shape: (2, ) tuple of (height, width) of the splatting grid
    returns:
        grid: (H, W, C) tensor with splatted data
        counts: (H, W, 1) tensor with splatted ones
    """
    assert data.device == data_coords.device
    device = data.device

    H, W = grid_shape[:2]
    C = data.shape[1]

    x = data_coords[:, 0]
    y = data_coords[:, 1]

    ## find the surrounding grid positions
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    # stay inside the grid
    x = torch.clamp(x, 0, W - 1)
    y = torch.clamp(y, 0, H - 1)
    x0 = torch.clamp(x0, 0, W - 1)
    x1 = torch.clamp(x1, 0, W - 1)
    y0 = torch.clamp(y0, 0, H - 1)
    y1 = torch.clamp(y1, 0, H - 1)

    # compute the bilinear coefficients
    x0f = x0.float()
    x1f = x1.float()
    y0f = y0.float()
    y1f = y1.float()
    w_a = einops.rearrange((x1f - x) * (y1f - y), 'N -> N 1')
    w_b = einops.rearrange((x1f - x) * (y - y0f), 'N -> N 1')
    w_c = einops.rearrange((x - x0f) * (y1f - y), 'N -> N 1')
    w_d = einops.rearrange((x - x0f) * (y - y0f), 'N -> N 1')

    # assert torch.all(w_a >= 0)
    # assert torch.all(w_b >= 0)
    # assert torch.all(w_c >= 0)
    # assert torch.all(w_d >= 0)
    # assert torch.all(w_a <= 1)
    # assert torch.all(w_b <= 1)
    # assert torch.all(w_c <= 1)
    # assert torch.all(w_d <= 1)

    row_indices = torch.cat((y0, y1, y0, y1), dim=0)
    col_indices = torch.cat((x0, x0, x1, x1), dim=0)
    flat_indices = torch_ravel_multi_index((row_indices, col_indices), (H, W))
    flat_data = torch.cat((data * w_a, data * w_b, data * w_c, data * w_d), dim=0)
    # flat_data = einops.rearrange(flat_data, 'N 1 -> N')
    flat_count = torch.cat((w_a, w_b, w_c, w_d), dim=0)

    grid_flat = torch.zeros((H * W, C), dtype=flat_data.dtype, device=device)
    grid_flat = grid_flat.index_put(indices=[flat_indices], values=flat_data, accumulate=True)

    counts_flat = torch.zeros((H * W, 1), dtype=flat_count.dtype, device=device)
    counts_flat = counts_flat.index_put(indices=[flat_indices], values=flat_count, accumulate=True)
    # data_a = data[:, y0, x0]
    # data_b = data[:, y1, x0]
    # data_c = data[:, y0, x1]
    # data_d = data[:, y1, x1]

    # interp = (w_a * data_a) + (w_b * data_b) + (w_c * data_c) + (w_d * data_d)
    grid = einops.rearrange(grid_flat, '(H W) C -> H W C', H=H, W=W, C=C)
    counts = einops.rearrange(counts_flat, '(H W) 1 -> H W 1', H=H, W=W)
    return grid, counts


def torch_ravel_multi_index(multi_index, dims):
    """Converts a tuple of index arrays into an array of flat indices.

    A counterpart to numpy ravel_multi_index.

    args:
        multi_index: a tuple of (N, ) integer tensors, one for each dimension
        dims: the shape of the array that multi_index indices refer to
    returns:
        raveled_indices: a (N, ) tensor of flat indices
    """
    if len(dims) != 2:
        raise NotImplementedError("Too lazy to do the most general thing... :)")
    H, W = dims
    rows = multi_index[0]
    cols = multi_index[1]

    raveled_indices = W * rows + cols
    return raveled_indices
