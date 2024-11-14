import os
import sys
from tempfile import NamedTemporaryFile
import logging
import einops
import numpy as np
import torch
from copy import copy
import torch.nn.functional as F
from MFTIQ.utils import interpolation
from MFTIQ.utils.geom_utils import torch_get_featuremap_coords, torch_get_featuremap_coords_cached
from MFTIQ.utils.misc import ensure_torch, ensure_numpy
import MFTIQ.utils.io as io_utils

logger = logging.getLogger(__name__)


class FlowOUTrackingResult(object):
    def __init__(self, flow, occlusion=None, sigma=None,
                 compressed_data=None, compressed_device=None):
        """Stores optical flow, occlusion map and flow sigma map.

        flow: (xy-delta, H, W) tensor
        occlusion: (1, H, W) tensor
        sigma: (1, H, W) tensor
        """
        if compressed_data is not None:
            self._compressed_data = compressed_data
            self._device = compressed_device
            self._flow = None
            self._occlusion = None
            self._sigma = None
            return

        assert len(flow.shape) == 3
        assert flow.shape[0] == 2
        self.H, self.W = flow.shape[1:]
        device = flow.device

        if occlusion is None:
            occlusion = torch.zeros((1, self.H, self.W), dtype=torch.float32, device=device)
        if sigma is None:
            sigma = torch.zeros((1, self.H, self.W), dtype=torch.float32, device=device)

        assert flow.shape == (2, self.H, self.W)
        assert occlusion.shape == (1, self.H, self.W)
        assert sigma.shape == (1, self.H, self.W)

        assert torch.all(occlusion >= 0)
        assert torch.all(occlusion <= 1.000001)
        assert torch.all(sigma >= 0)

        self._flow = flow  # (xy-delta, H, W) tensor
        self._occlusion = occlusion  # (1, H, W) tensor
        self._sigma = sigma  # (1, H, W) tensor

        self._compressed_data = None
        self._device = device

    @property
    def flow(self):
        self.decompress()
        return self._flow

    @property
    def occlusion(self):
        self.decompress()
        return self._occlusion

    @occlusion.setter
    def occlusion(self, value):
        compressed = self.compressed()

        if compressed:
            self.decompress()

        self._occlusion = value

        if compressed:
            self.compress()

    @property
    def sigma(self):
        self.decompress()
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        compressed = self.compressed()

        if compressed:
            self.decompress()

        self._sigma = value

        if compressed:
            self.compress()

    def compressed(self):
        compressed = self._compressed_data is not None
        # if compressed:
        #     assert self._flow is None
        # else:
        #     assert self._flow is not None
        return compressed

    def __repr__(self):
        cls = self.__class__.__name__
        compressed = '(compressed)' if self.compressed() else ''
        txt = f'<{cls} ({self.H} x {self.W}) has flow, occlusion, sigma {compressed}>'
        return txt

    def cpu(self):
        if not self.compressed():
            self._flow = self._flow.cpu()
            self._occlusion = self._occlusion.cpu()
            self._sigma = self._sigma.cpu()
        else:
            self._device = 'cpu'
        return self

    def cuda(self):
        if not self.compressed():
            self._flow = self._flow.cuda()
            self._occlusion = self._occlusion.cuda()
            self._sigma = self._sigma.cuda()
        else:
            self._device = 'cuda'
        return self

    @property
    def device(self):
        if not self.compressed():
            if self._flow.is_cuda:
                return 'cuda'
            else:
                return 'cpu'
        else:
            return self._device

    def clone(self):
        if not self.compressed():
            return FlowOUTrackingResult(self.flow.clone(),
                                        self.occlusion.clone(),
                                        self.sigma.clone())
        else:
            return FlowOUTrackingResult(None,
                                        None,
                                        None,
                                        compressed_data=copy(self.compressed_data))

    def write(self, path):
        io_utils.write_flowou(path,
                              ensure_numpy(self.flow),
                              ensure_numpy(self.occlusion),
                              ensure_numpy(self.sigma))

    @classmethod
    def read(self, path):
        flow, occlusions, sigmas = io_utils.read_flowou(path)
        return FlowOUTrackingResult(torch.from_numpy(flow),
                                    torch.from_numpy(occlusions),
                                    torch.from_numpy(sigmas))

    @classmethod
    def identity(cls, flow_shape, device=None):
        """Create a zero-flow, zero-sigma, zero-occlusion result

        args:
            flow_shape: (H, W) tuple"""
        xy = 2
        flow = torch.zeros((xy, flow_shape[0], flow_shape[1]), dtype=torch.float32, device=device)
        occlusion = torch.zeros((1, flow_shape[0], flow_shape[1]), dtype=torch.float32, device=device)
        sigma = torch.zeros((1, flow_shape[0], flow_shape[1]), dtype=torch.float32, device=device)

        return FlowOUTrackingResult(flow, occlusion=occlusion, sigma=sigma)

    def compress(self):
        """Compress the FlowOU in RAM."""
        if self.compressed():
            # already compressed
            return self

        # create a temporary file in /dev/shm which is a RAMdisk on linux
        tmp_file = NamedTemporaryFile(delete=False, suffix=".flowouX16.pkl", dir="/dev/shm")
        tmp_file.close()

        io_utils.write_flowou_X16(tmp_file.name,
                                  ensure_numpy(self._flow),
                                  ensure_numpy(self._occlusion),
                                  ensure_numpy(self._sigma))

        with open(tmp_file.name, 'rb') as fin:
            self._compressed_data = fin.read()

        self._device = self._flow.device
        self._flow = None
        self._occlusion = None
        self._sigma = None

        os.unlink(tmp_file.name)
        return self

    def decompress(self):
        if not self.compressed():
            # already decompressed
            return

        logger.debug('Decompressing FlowOU')

        # create a temporary file in /dev/shm which is a RAMdisk on linux
        tmp_file = NamedTemporaryFile(delete=False, suffix=".flowouX16.pkl", dir="/dev/shm")
        tmp_file.write(self._compressed_data)
        tmp_file.close()

        flow, occlusion, sigma = io_utils.read_flowou_X16(tmp_file.name)
        self._flow = torch.from_numpy(flow).to(self._device)
        self._occlusion = torch.from_numpy(occlusion).to(self._device)
        self._sigma = torch.from_numpy(sigma).to(self._device)
        self._compressed_data = None

        os.unlink(tmp_file.name)
        return self

    def used_memory(self):
        if self.compressed():
            return 0
        else:
            return (sys.getsizeof(self.flow.untyped_storage()) +
                    sys.getsizeof(self.occlusion.untyped_storage()) +
                    sys.getsizeof(self.sigma.untyped_storage()))

    def chain(self, flow):
        """Chain 'flow' after our result.

        With 'self.flow' from A to B, and 'flow' from B to C, this
        function produces a flow from A to C by bilinear
        interpolation.

        args:
            flow: (xy H W) tensor with flow from B to C
        return:
            chained_flow: (xy H W) tensor with flow from A to C
        """
        device = flow.device
        flow_shape = einops.parse_shape(flow, 'xy H W')
        assert flow_shape['xy'] == 2
        coords_A = torch_get_featuremap_coords((flow_shape['H'], flow_shape['W']),
                                               device=device, keep_shape=True)
        coords_B = coords_A + self.flow.to(device).to(torch.float32)
        coords_B_normed = interpolation.normalize_coords(einops.rearrange(coords_B, 'xy H W -> 1 H W xy', xy=2),
                                                         flow_shape['H'], flow_shape['W'])

        flow_BC = flow.to(torch.float32)
        flow_BC_sampled = F.grid_sample(einops.rearrange(flow_BC, 'xy H W -> 1 xy H W', xy=2),
                                        coords_B_normed,
                                        align_corners=True)
        flow_AC = coords_B + flow_BC_sampled - coords_A
        chained_flow = einops.rearrange(flow_AC, '1 xy H W -> xy H W', xy=2)
        return chained_flow

    def warp_backward(self, img):
        """Sample the img data using the right end of 'self.flow'

        args:
            img: (C, H, W) tensor

        return:
            img_sampled: (C, flow_H, flow_W) tensor
        """
        assert len(img.shape) == 3
        assert img.shape[1:] == (self.H, self.W)
        device = img.device
        coords_A = torch_get_featuremap_coords((self.H, self.W), device=device, keep_shape=True)
        coords_B = coords_A + self.flow.to(device).to(torch.float32)
        coords_B_normed = interpolation.normalize_coords(einops.rearrange(coords_B, 'xy H W -> 1 H W xy', xy=2),
                                                         self.H, self.W)

        img_sampled = F.grid_sample(einops.rearrange(img, 'C H W -> 1 C H W'),
                                    coords_B_normed,
                                    align_corners=True)
        return einops.rearrange(img_sampled, '1 C H W -> C H W', H=self.H, W=self.W)

    def warp_forward_points(self, points):
        """Warp the points using the stored optical flow.

        args:
            points: (N xy) tensor with source coordinates.
        return:
            warped_points: (N xy) tensor with warped coordinates
        """
        points = ensure_torch(points).to(torch.float32)
        N = points.shape[0]
        device = points.device
        # (we will keep N points in the usual Width dimension and keep Height = 1)
        points_normed = interpolation.normalize_coords(einops.rearrange(points, 'N xy -> 1 1 N xy', xy=2),
                                                       H=self.H, W=self.W)

        flow = self.flow.to(device).to(torch.float32)
        flow_sampled = F.grid_sample(einops.rearrange(flow, 'xy H W -> 1 xy H W', xy=2),
                                     points_normed, align_corners=True)
        warped_points = points + einops.rearrange(flow_sampled, '1 xy 1 N -> N xy', xy=2, N=N)
        return warped_points

    def sample(self, points):
        """Sample the results at the given points.

        args:
            points: (N xy) tensor with query coordinates
        return:
            sampled_flow: (xy N)
            sampled_occlusions: (1 N)
            sampled_sigmas: (1 N)
        """
        points = ensure_torch(points).to(torch.float32)
        N = points.shape[0]
        device = points.device
        # (we will keep N points in the usual Width dimension and keep Height = 1)
        points_normed = interpolation.normalize_coords(einops.rearrange(points, 'N xy -> 1 1 N xy', xy=2),
                                                       H=self.H, W=self.W)

        flow = self.flow.to(device).to(torch.float32)
        sampled_flow = F.grid_sample(einops.rearrange(flow, 'xy H W -> 1 xy H W', xy=2),
                                     points_normed, align_corners=True)
        sampled_flow = einops.rearrange(sampled_flow, '1 xy 1 N -> xy N', xy=2, N=N)
        sampled_occlusions = F.grid_sample(
            einops.rearrange(self.occlusion.to(device).to(torch.float32), '1 H W -> 1 1 H W'),
            points_normed, align_corners=True)
        sampled_occlusions = einops.rearrange(sampled_occlusions, '1 c 1 N -> c N', c=1, N=N)
        sampled_sigmas = F.grid_sample(
            einops.rearrange(self.sigma.to(device).to(torch.float32), '1 H W -> 1 1 H W'),
            points_normed, align_corners=True)
        sampled_sigmas = einops.rearrange(sampled_sigmas, '1 c 1 N -> c N', c=1, N=N)
        return sampled_flow, sampled_occlusions, sampled_sigmas

    def warp_forward(self, img, mask=None, border=None, to_numpy=True):
        """Warp img forward by self.flow, only warp values that are not masked.

        args:
            img: (H, W, ...) array / tensor with values to be warped
            mask: [optional] (H, W) binary array / tensor. Only values with mask==True will be warped
            border: [optional] border value

        returns:
            warped_img: (H, W, ...) array with warped values
        """
        device = self.flow.device
        assert img.shape[:2] == (self.H, self.W)

        H, W = img.shape[:2]
        xs, ys = np.meshgrid(np.arange(W), np.arange(H))
        assert xs.shape == (H, W)
        assert ys.shape == (H, W)
        assert xs[0, 0] == xs[1, 0]

        flat_img = False
        if len(img.shape) == 2:
            flat_img = True
            img = einops.rearrange(img, 'H W -> H W 1')

        # debugging
        vmin, vmax = img.min(), img.max()

        src_coords = np.dstack((xs, ys))
        query_coords_flat = einops.rearrange(src_coords, 'H W xy -> (H W) xy')

        dst_coords = torch.from_numpy(src_coords).to(device) + einops.rearrange(self.flow,
                                                                                'xy H W -> H W xy',
                                                                                xy=2, H=H, W=W)
        pixel_positions_flat = einops.rearrange(dst_coords, 'H W xy -> (H W) xy', xy=2)
        pixel_values_flat = einops.rearrange(img, 'H W ... -> (H W) ...')
        if mask is not None:
            pixel_mask = mask[query_coords_flat[:, 1], query_coords_flat[:, 0]]
            pixel_positions_flat = pixel_positions_flat[pixel_mask, :]
            pixel_values_flat = ensure_torch(pixel_values_flat).to(device)[pixel_mask, ...]

        pixel_values_flat = ensure_torch(pixel_values_flat)
        assert torch.all(pixel_values_flat >= vmin)
        assert torch.all(pixel_values_flat <= vmax)

        accum, counts = interpolation.bilinear_splat(pixel_values_flat.to(pixel_positions_flat.device),
                                                     pixel_positions_flat,
                                                     (H, W))
        warped_img = accum.clone()
        count_mask = einops.rearrange(counts, 'H W 1 -> H W') > 0
        warped_img[count_mask] /= counts[count_mask]

        out_vmin = warped_img.min()
        out_vmax = warped_img.max()

        eps = 5e-3
        # errors = output < (min(vmin, 0) - eps)
        assert torch.all(out_vmin >= min(vmin, 0) - eps)  # 0 for the places without any data
        assert torch.all(out_vmax <= max(vmax, 0) + eps)

        if border is not None:
            warped_img[~count_mask] = border

        if to_numpy:
            warped = warped_img.cpu().numpy()
        else:
            warped = warped_img
        if flat_img:
            warped = einops.rearrange(warped, 'H W 1 -> H W')
        return warped


    def invalid_mask(self):
        """Compute a mask of invalid self.flows, i.e. pointing outside of the image.

        returns:
            invalid_mask: (H, W) bool tensor, with True meaning the flow is invalid
        """
        device = self.flow.device
        coords_A = torch_get_featuremap_coords((self.H, self.W), device=device, keep_shape=True)
        coords_B = coords_A + self.flow.to(torch.float32)  # xy, H, W

        invalid_mask = torch.logical_or(
            torch.any(coords_B < 0, dim=0),
            torch.logical_or(
                coords_B[0, :, :] >= self.W,
                coords_B[1, :, :] >= self.H))
        return invalid_mask

    def get_end_coords(self, use_cache=False):
        """Compute a tensor of absolute coordinates (flow + reference coordinate)

        returns:
            coords_B: (xy, H, W) tensor with the coordinates in the second image
        """
        device = self.flow.device
        if use_cache:
            fn = torch_get_featuremap_coords_cached
        else:
            fn = torch_get_featuremap_coords # type: ignore
        coords_A = fn((self.H, self.W), device=device, keep_shape=True)
        coords_B = coords_A + self.flow.to(torch.float32)  # xy, H, W
        return coords_B

    def get_start_coords(self, use_cache=False):
        """Compute a tensor of reference coordinates

        returns:
            coords_A: (xy, H, W) tensor with the coordinates in the first
        """
        device = self.flow.device
        if use_cache:
            fn = torch_get_featuremap_coords_cached
        else:
            fn = torch_get_featuremap_coords # type: ignore
        coords_A = fn((self.H, self.W), device=device, keep_shape=True)
        return coords_A
