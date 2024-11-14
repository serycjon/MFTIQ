import sys
import glob
import time
import datetime
from pathlib import Path
from collections import deque, Counter
import cv2
import re
import gzip
from tempfile import NamedTemporaryFile
import pickle
import io
import numpy as np
import einops
import os
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import shutil
from collections import OrderedDict
from MFTIQ.utils.misc import ensure_numpy
import tqdm
from MFTIQ.utils.misc import dummy_profile
import multiprocessing
import queue
from itertools import cycle
import lz4.frame
from MFTIQ.UOM.utils.image_manipulation import RotateAndPad

import logging
logger = logging.getLogger(__name__)
profile = dummy_profile()


def get_frames(path):
    paths = glob.glob(f'{path}/*.jpg')
    return sorted([Path(path) for path in paths])


def video_seek_frame(time_string, fps=30):
    parsed_time = time.strptime(time_string, "%H:%M:%S")
    delta = datetime.timedelta(hours=parsed_time.tm_hour, minutes=parsed_time.tm_min, seconds=parsed_time.tm_sec)
    time_seconds = int(delta.total_seconds())
    pos = fps * time_seconds
    return pos


def video_seek_frame_name(query_frame_name, frame_paths):
    frame_names = [path.stem for path in frame_paths]
    regexp = re.compile(r'0*' + query_frame_name)
    for i, name in enumerate(frame_names):
        if re.match(regexp, name):
            return i
    raise ValueError(f"Frame {query_frame_name} not found.")


def frames_from_time(directory, time_string, fps=30):
    frames = get_frames(directory)
    start_index = video_seek_frame(time_string, fps)

    for i in range(start_index, len(frames)):
        yield (frames[i], cv2.imread(str(frames[i])))


def frames_from_name(directory, start_name):
    frames = get_frames(directory)
    start_index = video_seek_frame_name(start_name, frames)

    for i in range(start_index, len(frames)):
        yield (frames[i], cv2.imread(str(frames[i])))


class LookaheadIter:

    def __init__(self, it):
        self._iter = iter(it)
        self._ahead = deque()

    def __iter__(self):
        return self

    def __next__(self):
        if self._ahead:
            return self._ahead.popleft()
        else:
            return next(self._iter)

    def lookahead(self):
        for x in self._ahead:
            yield x
        for x in self._iter:
            self._ahead.append(x)
            yield x

    def peek(self, *a):
        return next(iter(self.lookahead()), *a)


def load_maybe_gzipped_pkl(path):
    suffix = Path(path).suffix
    if suffix == '.pklz':
        open_fn = gzip.open
    elif suffix == '.pkl':
        open_fn = open
    else:
        ValueError(f"Unknown pickle file suffix ({suffix}).")

    with open_fn(path, 'rb') as fin:
        data = pickle.load(fin)

    return data


class CPU_Unpickler(pickle.Unpickler):
    """ https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219
    I have pickled something in meta as a GPU tensor..."""

    def find_class(self, module, name):
        import torch

        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def load_cpu_pickle(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"No pickle at {path}")
    try:
        exception = gzip.BadGzipFile  # new in python 3.8
    except AttributeError:
        exception = OSError

    try:
        with gzip.open(path, 'rb') as fin:
            unpickler = CPU_Unpickler(fin)
            data = unpickler.load()
    except exception:  # we didn't compress this one...
        with open(path, 'rb') as fin:
            unpickler = CPU_Unpickler(fin)
            data = unpickler.load()
    return data


def read_flow_png(path):
    """Read png-compressed flow

    Args:
        path: png flow file path

    Returns:
        flow: (H, W, 2) float32 numpy array (delta-x, delta-y)
        valid: (H, W) float32 numpy array
    """
    # to specify not to change the image depth (16bit)
    flow = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    flow = flow[:, :, ::-1].astype(np.float32)
    # flow shape (H, W, 2) valid shape (H, W)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 32.0
    return flow, valid


def write_flow_png(path, flow, valid=None):
    """Write a compressed png flow

    Args:
        path: write path
        flow: (H, W, 2) xy-flow
        valid: None, or (H, W) array with validity mask
    """
    flow = 32.0 * flow + 2**15  # compress (resolution step 1/32, maximal flow 1024 (same as Sintel width))
    if valid is None:
        valid = np.ones([flow.shape[0], flow.shape[1], 1])
    else:
        valid = einops.rearrange(valid, 'H W -> H W 1', **einops.parse_shape(flow, 'H W _'))
    data = np.concatenate([flow, valid], axis=2).astype(np.uint16)
    cv2.imwrite(str(path), data[:, :, ::-1])


# flow is encoded with sign, so 2**15, occlusion and uncertainty without sign, so 2**16:
FLOWOU_IO_FLOW_MULTIPLIER = 2**5  # max-abs-val = 2**(15-5) = 1024, step = 2**(-5) = 0.03
FLOWOU_IO_OCCLUSION_MULTIPLIER = 2**15  # max-val = 2**(16-15) = 2, step = 2**(-15) = 3e-5
FLOWOU_IO_UNCERTAINTY_MULTIPLIER = 2**9   # max-val = 2**(16-9) = 128, step = 2**(-9) = 0.0019


def write_flowou(path, flow, occlusions, uncertainty, meta=None):
    """Write a compressed png flow, occlusions and uncertainty

    Args:
        path: write path (must have ".flowou.png", ".flowouX16.pkl", or ".flowouX32.pkl" suffix)
        flow: (2, H, W) xy-flow
        occlusions: (1, H, W) array with occlusion scores (1 = occlusion, 0 = visible)
        uncertainty: (1, H, W) array with uncertainty sigma
        meta: [optional] a dictionary with metadata
    """
    if occlusions is None:
        occlusions = np.zeros((1, flow.shape[0], flow.shape[1]), dtype=np.float32)
    if uncertainty is None:
        uncertainty = np.zeros((1, flow.shape[0], flow.shape[1]), dtype=np.float32)

    suf = Path(path).suffixes[0]
    if suf == '.flowou':
        write_flowou1_png(path, flow, occlusions, uncertainty, meta)
    elif suf == '.flowouX16':
        write_flowou_X16(path, flow, occlusions, uncertainty, meta)
    elif suf == '.flowouX32':
        write_flowou_X32(path, flow, occlusions, uncertainty, meta)
    elif suf == '.stepan16':
        write_flowou_stepan16(path, flow, occlusions, uncertainty, meta)
    else:
        raise ValueError(f"Incorrect flowou path suffix: {Path(path).suffixes}")


@profile
def read_flowou(path, return_meta=False):
    """Read png-compressed flow, occlusions and uncertainty

    Args:
        path: ".flowou.png", ".flowouX16.pkl", or ".flowouX32.pkl" file path
        return_meta: [optional] return metadata

    Returns:
        flow: (2, H, W) float32 numpy array (delta-x, delta-y)
        occlusions: (1, H, W) float32 array with occlusion scores (1 = occlusion, 0 = visible)
        uncertainty: (1, H, W) float32 array with uncertainty sigma (0 = dirac)
        meta: [only when return_meta=True] metadata dict
    """
    suf = Path(path).suffixes[0]
    if suf == '.flowou':
        return read_flowou1_png(path, return_meta)
    elif suf == '.flowouX16':
        return read_flowou_X16(path, return_meta)
    elif suf == '.flowouX32':
        return read_flowou_X32(path, return_meta)
    else:
        raise ValueError(f"Incorrect flowou path suffix: {Path(path).suffixes}")


def write_flowou1_png(path, flow, occlusions, uncertainty, meta=None):
    """Write a compressed png flow, occlusions and uncertainty

    Args:
        path: write path (must have ".flowou.png" suffix)
        flow: (2, H, W) xy-flow
        occlusions: (1, H, W) array with occlusion scores (1 = occlusion, 0 = visible), clipped between 0 and 1
        uncertainty: (1, H, W) array with uncertainty sigma, clipped between 0 and 2047
                      (0 = dirac, max observed on Sintel = 215, Q0.999 on sintel ~ 15)
    """
    if meta is not None:
        raise ValueError("The .flowou.png format is incompatible with metadata saving.")

    def encode_central(xs, multiplier=32.0):
        max_val = 2**15 / multiplier
        assert np.all(np.abs(xs) < max_val), "out-of-range values - cannot be written"
        return 2**15 + multiplier * xs

    def encode_positive(xs, multiplier=32.0):
        max_val = 2**16 / multiplier
        assert np.all(xs >= 0), "out-of-range values - cannot be written"
        assert np.all(xs < max_val), "out-of-range values - cannot be written"
        return multiplier * xs

    assert Path(path).suffixes == ['.flowou', '.png']
    path.parent.mkdir(parents=True, exist_ok=True)
    einops.parse_shape(flow, 'H W xy')
    flow = encode_central(einops.rearrange(flow, 'xy H W -> H W xy', xy=2),
                          multiplier=FLOWOU_IO_FLOW_MULTIPLIER)

    occlusions = np.clip(occlusions, 0, 1)
    occlusions = encode_positive(einops.rearrange(occlusions, '1 H W -> H W 1', **einops.parse_shape(flow, 'H W _')),
                                 multiplier=FLOWOU_IO_OCCLUSION_MULTIPLIER)

    uncertainty = np.clip(uncertainty, 0, 127)
    uncertainty = encode_positive(einops.rearrange(uncertainty, '1 H W -> H W 1', **einops.parse_shape(flow, 'H W _')),
                                  multiplier=FLOWOU_IO_UNCERTAINTY_MULTIPLIER)

    data = np.concatenate([flow, occlusions, uncertainty], axis=2).astype(np.uint16)
    cv2.imwrite(str(path), data)


def read_flowou1_png(path, return_meta=False):
    """Read png-compressed flow, occlusions and uncertainty

    Args:
        path: ".flowou.png" file path

    Returns:
        flow: (2, H, W) float32 numpy array (delta-x, delta-y)
        occlusions: (1, H, W) float32 array with occlusion scores (1 = occlusion, 0 = visible)
        uncertainty: (1, H, W) float32 array with uncertainty sigma (0 = dirac)
    """
    if return_meta:
        raise ValueError("The .flowou.png format cannot store metadata.")
    # to specify not to change the image depth (16bit)
    assert Path(path).suffixes == ['.flowou', '.png']

    def decode_central(xs, multiplier=32.0):
        return (xs.astype(np.float32) - 2**15) / multiplier

    def decode_positive(xs, multiplier=32.0):
        return xs.astype(np.float32) / multiplier

    data = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
    data = einops.rearrange(data, 'H W C -> C H W', C=4)
    flow, occlusions, uncertainty = data[:2, :, :], data[2, :, :], data[3, :, :]
    occlusions = einops.rearrange(occlusions, 'H W -> 1 H W')
    uncertainty = einops.rearrange(uncertainty, 'H W -> 1 H W')
    flow = decode_central(flow, multiplier=FLOWOU_IO_FLOW_MULTIPLIER)
    occlusions = decode_positive(occlusions, multiplier=FLOWOU_IO_OCCLUSION_MULTIPLIER)
    uncertainty = decode_positive(uncertainty, multiplier=FLOWOU_IO_UNCERTAINTY_MULTIPLIER)
    return flow, occlusions, uncertainty


def write_flowou2_png(path, flow, occlusions, uncertainty):
    """Write a compressed png flow, occlusions and uncertainty, with a variable min-max range

    Args:
        path: write path (must have ".flowou2.png" suffix)
        flow: (2, H, W) xy-flow
        occlusions: (1, H, W) array with occlusion scores (1 = occlusion, 0 = visible), clipped between 0 and 1
        uncertainty: (1, H, W) array with uncertainty sigma, clipped between 0 and 2047
                      (0 = dirac, max observed on Sintel = 215, Q0.999 on sintel ~ 15)
    """
    def encode(xs):
        f_xs = np.float32(xs)
        lb = np.amin(f_xs)
        ub = np.amax(f_xs)

        if np.abs(ub - lb) < 1e-8:
            xs_01 = np.zeros_like(f_xs)
        else:
            xs_01 = (f_xs - lb) / (ub - lb)

        uint16_xs = np.uint16(xs_01 * (2**16 - 1))
        return uint16_xs, lb, ub

    assert Path(path).suffixes == ['.flowou2', '.png']
    path.parent.mkdir(parents=True, exist_ok=True)
    einops.parse_shape(flow, 'H W xy')
    flow, flow_min, flow_max = encode(einops.rearrange(flow, 'xy H W -> H W xy', xy=2))

    occlusions, occl_min, occl_max = encode(einops.rearrange(occlusions, '1 H W -> H W 1',
                                                             **einops.parse_shape(flow, 'H W _')))

    uncertainty, unc_min, unc_max = encode(einops.rearrange(uncertainty, '1 H W -> H W 1',
                                                            **einops.parse_shape(flow, 'H W _')))

    data = np.concatenate([flow, occlusions, uncertainty], axis=2)
    pil_data = Image.fromarray(data)
    metadata = PngInfo()
    metadata.add_text("flow_min", str(flow_min))
    metadata.add_text("flow_max", str(flow_max))

    metadata.add_text("occl_min", str(occl_min))
    metadata.add_text("occl_max", str(occl_max))

    metadata.add_text("unc_min", str(unc_min))
    metadata.add_text("unc_max", str(unc_max))
    pil_data.save(str(path), pnginfo=metadata)


def read_flowou2_png(path):
    """Read png-compressed flow, occlusions and uncertainty, with a variable min-max range

    Args:
        path: ".flowou2.png" file path

    Returns:
        flow: (2, H, W) float32 numpy array (delta-x, delta-y)
        occlusions: (1, H, W) float32 array with occlusion scores (1 = occlusion, 0 = visible)
        uncertainty: (1, H, W) float32 array with uncertainty sigma (0 = dirac)
    """
    # to specify not to change the image depth (16bit)
    assert Path(path).suffixes == ['.flowou2', '.png']

    def decode(xs, lb, ub):
        xs_01 = np.float32(xs) / (2**16 - 1)
        return lb + xs_01 * (ub - lb)

    pil_data = Image.open(str(path))
    metadata = pil_data.text
    data = np.asarray(pil_data)
    data = einops.rearrange(data, 'H W C -> C H W', C=4)
    flow, occlusions, uncertainty = data[:2, :, :], data[2, :, :], data[3, :, :]
    occlusions = einops.rearrange(occlusions, 'H W -> 1 H W')
    uncertainty = einops.rearrange(uncertainty, 'H W -> 1 H W')
    flow = decode(flow, float(metadata['flow_min']), float(metadata['flow_max']))
    occlusions = decode(occlusions, float(metadata['occl_min']), float(metadata['occl_max']))
    uncertainty = decode(uncertainty, float(metadata['unc_min']), float(metadata['unc_max']))
    return flow, occlusions, uncertainty


def write_flowou_X32(path, flow, occlusions, uncertainty, meta=None):
    def compress_channel(xs):
        f_xs = np.float32(xs)
        lb = np.nanmin(f_xs)
        ub = np.nanmax(f_xs)

        if np.abs(ub - lb) < 1e-8:
            xs_01 = np.zeros_like(f_xs)
        else:
            xs_01 = (f_xs - lb) / (ub - lb)
        uint32_xs = np.uint32(xs_01 * (2**32 - 1))
        return uint32_xs, lb, ub

    def u32_to_4u8(xs):
        assert len(xs.shape) == 2, f"Need a HxW array, got {xs.shape} instead"
        byte_1 = np.uint8(xs & 0x000000FF)
        byte_2 = np.uint8((xs & 0x0000FF00) >> 8)
        byte_3 = np.uint8((xs & 0x00FF0000) >> 16)
        byte_4 = np.uint8((xs & 0xFF000000) >> 24)
        return np.dstack((byte_4, byte_3, byte_2, byte_1))

    def encode_channel(xs):
        compressed_xs, lb, ub = compress_channel(xs)
        xs_4u8 = u32_to_4u8(compressed_xs)
        is_success, buf = cv2.imencode(".png", xs_4u8)
        # https://stackoverflow.com/a/52865864/1705970
        return {'data': buf,
                'min': lb,
                'max': ub}

    result = {
        'flow_x': encode_channel(flow[0, :, :]),
        'flow_y': encode_channel(flow[1, :, :]),
        'occlusion': encode_channel(occlusions[0, :, :]),
        'sigma': encode_channel(uncertainty[0, :, :])
        }

    if meta is not None:
        result['meta'] = meta

    with open(path, 'wb') as fout:
        pickle.dump(result, fout)


def read_flowou_X32(path, return_meta=False):
    def decode_channel(data):
        buf = data['data']
        # https://stackoverflow.com/a/52865864/1705970
        xs_4u8 = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        xs_compressed = data_4u8_to_u32(xs_4u8)
        xs = decompress_channel(xs_compressed, data['min'], data['max'])
        return xs

    def data_4u8_to_u32(xs):
        assert xs.dtype == np.uint8
        byte_4, byte_3, byte_2, byte_1 = np.dsplit(np.uint32(xs), 4)

        u32 = (byte_4 << 24) | (byte_3 << 16) | (byte_2 << 8) | byte_1
        return einops.rearrange(u32, 'H W 1 -> H W')

    def decompress_channel(compressed_xs, lb, ub):
        xs_01 = np.float32(compressed_xs) / (2**32 - 1)
        xs = (xs_01 * (ub - lb)) + lb
        return xs

    with open(path, 'rb') as fin:
        data = pickle.load(fin)

    flow_x = decode_channel(data['flow_x'])
    flow_y = decode_channel(data['flow_y'])
    flow = np.stack((flow_x, flow_y), axis=0)
    uncertainty = einops.rearrange(decode_channel(data['sigma']), 'H W -> 1 H W')
    occlusions = einops.rearrange(decode_channel(data['occlusion']), 'H W -> 1 H W')

    if return_meta:
        meta = data.get('meta', {})
        return flow, occlusions, uncertainty, meta
    else:
        return flow, occlusions, uncertainty

    return flow, occlusions, uncertainty


def write_flowou_stepan16(path, flow, occlusions, uncertainty, meta=None):
    if meta is not None:
        raise ValueError("The .stepan16 format is incompatible with metadata saving.")

    def compress_channel(xs):
        f_xs = np.float32(xs)
        lb = np.amin(f_xs)
        ub = np.amax(f_xs)

        if np.abs(ub - lb) < 1e-8:
            xs_01 = np.zeros_like(f_xs)
        else:
            xs_01 = (f_xs - lb) / (ub - lb)
        uint16_xs = np.uint16(np.round(xs_01 * (2**16 - 1)))
        return uint16_xs, lb, ub

    def u16_to_3u8(xs):
        assert len(xs.shape) == 2, f"Need a HxW array, got {xs.shape} instead"
        byte_1 = np.uint8(xs & 0x00FF)
        byte_2 = np.uint8((xs & 0xFF00) >> 8)
        byte_3 = np.uint8((xs & 0x0000) >> 16)
        return np.dstack((byte_3, byte_2, byte_1))

    def encode_channel(xs):
        compressed_xs, lb, ub = compress_channel(xs)
        xs_3u8 = u16_to_3u8(compressed_xs)
        is_success, buf = cv2.imencode(".png", xs_3u8)
        return {'data': buf,
                'min': lb,
                'max': ub}

    result = {
        'flow_x': encode_channel(flow[0, :, :]),
        'flow_y': encode_channel(flow[1, :, :]),
        }

    path = str(path)
    suffix = '.stepan16'
    assert path.endswith(suffix)
    path = path[:-len(suffix)]

    flow_x_path = path + '_flow_x.png'
    flow_y_path = path + '_flow_y.png'
    with open(flow_x_path, 'wb') as fout:
        fout.write(result['flow_x']['data'])
    with open(flow_y_path, 'wb') as fout:
        fout.write(result['flow_y']['data'])
    limits_path = path + '_limits.txt'
    with open(limits_path, 'w') as fout:
        fout.write(f"{result['flow_x']['min']} {result['flow_x']['max']} {result['flow_y']['min']} {result['flow_y']['max']}")


def write_flowou_X16(path, flow, occlusions, uncertainty, meta=None, encoding='lz4'):
    def compress_channel(xs):
        f_xs = np.float32(xs)
        lb = np.nanmin(f_xs)
        ub = np.nanmax(f_xs)
        if not np.all(np.isfinite(f_xs)):
            raise ValueError("Values contain NaNs or infs.")

        if np.abs(ub - lb) < 1e-8:
            xs_01 = np.zeros_like(f_xs)
        else:
            xs_01 = (f_xs - lb) / (ub - lb)

        if not np.all(np.isfinite(xs_01)):
            raise OverflowError("Values contain NaNs or infs after normalization to 0-1.")

        uint16_xs = np.uint16(np.round(xs_01 * (2**16 - 1)))
        return uint16_xs, lb, ub

    def u16_to_3u8(xs):
        assert len(xs.shape) == 2, f"Need a HxW array, got {xs.shape} instead"
        byte_1 = np.uint8(xs & 0x00FF)
        byte_2 = np.uint8((xs & 0xFF00) >> 8)
        byte_3 = np.uint8((xs & 0x0000) >> 16)
        return np.dstack((byte_3, byte_2, byte_1))

    def encode_channel(xs, encoding):
        compressed_xs, lb, ub = compress_channel(xs)
        height, width = compressed_xs.shape
        if encoding == 'lz4':
            comp_bytes = compressed_xs.tobytes()
            buf = lz4.frame.compress(comp_bytes)
        elif encoding == 'png':
            xs_3u8 = u16_to_3u8(compressed_xs)
            is_success, buf = cv2.imencode(".png", xs_3u8)
        else:
            raise NotImplementedError(f"Encoding {encoding} not implemented")
        return {'data': buf,
                'min': lb,
                'max': ub,
                'width': width,
                'height': height,
                }

    assert encoding == 'lz4' or encoding == 'png'
    result = {
        'flow_x': encode_channel(flow[0, :, :], encoding),
        'flow_y': encode_channel(flow[1, :, :], encoding),
        'occlusion': encode_channel(occlusions[0, :, :], encoding),
        'sigma': encode_channel(uncertainty[0, :, :], encoding),
        'encoding': encoding,
        }

    if meta is not None:
        result['meta'] = meta

    with open(path, 'wb') as fout:
        pickle.dump(result, fout)


def write_compress_X16(path, meta=None, encoding='lz4', **kwargs):
    """
    save everything from "kwargs"
    """
    def compress_channel(xs):
        f_xs = np.float32(xs)
        lb = np.nanmin(f_xs)
        ub = np.nanmax(f_xs)
        if not np.all(np.isfinite(f_xs)):
            raise ValueError("Values contain NaNs or infs.")

        if np.abs(ub - lb) < 1e-8:
            xs_01 = np.zeros_like(f_xs)
        else:
            xs_01 = (f_xs - lb) / (ub - lb)

        if not np.all(np.isfinite(xs_01)):
            raise OverflowError("Values contain NaNs or infs after normalization to 0-1.")

        uint16_xs = np.uint16(np.round(xs_01 * (2**16 - 1)))
        return uint16_xs, lb, ub

    def u16_to_3u8(xs):
        assert len(xs.shape) == 2, f"Need a HxW array, got {xs.shape} instead"
        byte_1 = np.uint8(xs & 0x00FF)
        byte_2 = np.uint8((xs & 0xFF00) >> 8)
        byte_3 = np.uint8((xs & 0x0000) >> 16)
        return np.dstack((byte_3, byte_2, byte_1))

    def encode_channel(xs, encoding):
        if xs.dtype == 'float32' or xs.dtype == 'float64' or xs.dtype == 'float16' or xs.dtype == 'float':
            compressed_xs, lb, ub = compress_channel(xs)
        else:
            compressed_xs = xs
            lb, ub = 0, 1

        height, width = compressed_xs.shape
        if encoding == 'lz4':
            comp_bytes = compressed_xs.tobytes()
            buf = lz4.frame.compress(comp_bytes)
        elif encoding == 'png':
            xs_3u8 = u16_to_3u8(compressed_xs)
            is_success, buf = cv2.imencode(".png", xs_3u8)
        else:
            raise NotImplementedError(f"Encoding {encoding} not implemented")
        return {'data': buf,
                'min': lb,
                'max': ub,
                'width': width,
                'height': height,
                'dtype': str(xs.dtype),
                }

    assert encoding == 'lz4' or encoding == 'png'
    result = {}
    for k, v in kwargs.items():
        result[k] = [encode_channel(v[i], encoding) for i in range(v.shape[0])]
    result['encoding'] = encoding

    if meta is not None:
        result['meta'] = meta

    with open(path, 'wb') as fout:
        pickle.dump(result, fout)


def read_compress_X16(path, return_meta=False):

    def decode_channel(data, encoding):
        buf = data['data']
        if encoding == 'lz4':
            if data['dtype'] in ['float32', 'float64', 'float16', 'float']:
                xs_u16 = np.frombuffer(lz4.frame.decompress(buf), dtype=np.uint16)
            else:
                xs_u16 = np.frombuffer(lz4.frame.decompress(buf), dtype=np.dtype(data['dtype']))
            xs_compressed = einops.rearrange(xs_u16, '(H W) -> H W', H=data['height'], W=data['width'])
        elif encoding == 'png':
            ## Faster decoding of PNGs, but getting Illegal hardware instruction coredump on Radon {{{
            # try:
            #     xs_3u8 = wuff_decode(buf)
            # except ImportError:
            #     warnings.warn("PyWuffs (fast PNG decoder) could not be imported, falling back to OpenCV")
            # }}}
            xs_3u8 = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            xs_compressed = data_3u8_to_u16(xs_3u8)
        else:
            raise NotImplementedError(f"Encoding {encoding} not implemented")
        if data['dtype'] in ['float32', 'float64', 'float16', 'float']:
            xs = decompress_channel(xs_compressed, data['min'], data['max'])
        else:
            xs = xs_compressed
        return xs

    def data_3u8_to_u16(xs):
        assert xs.dtype == np.uint8
        byte_3, byte_2, byte_1 = np.dsplit(np.uint16(xs), 3)

        u16 = (byte_2 << 8) | byte_1
        return einops.rearrange(u16, 'H W 1 -> H W')

    def decompress_channel(compressed_xs, lb, ub):
        if not np.all(np.isfinite((lb, ub))):
            raise ValueError(f"FlowOU compression lower bound ({lb}) or upper bound ({ub}) invalid.")
        xs_01 = np.float32(compressed_xs) / (2**16 - 1)
        xs = (xs_01 * (ub - lb)) + lb
        if not np.all(np.isfinite(xs)):
            raise ValueError("FlowOU contains NaNs or infs.")
        return xs

    with open(path, 'rb') as fin:
        data = pickle.load(fin)

    encoding = data['encoding'] if 'encoding' in data else 'png'
    assert encoding == 'png' or encoding == 'lz4'

    data_out = {}
    for k, v in data.items():
        if k == 'encoding':
            continue
        c_data = [decode_channel(cc, encoding).astype(cc['dtype']) for cc in v]
        data_out[k] = np.stack(c_data, axis=0)

    if return_meta:
        meta = data.get('meta', {})
        return data_out, meta
    else:
        return data_out




def write_normals(path, normals, mask=None, meta=None):
    """Write a compressed png normals and mask
    Function uses existing infrastructure of write_flow* functions for saving normals

    Args:
        path: write path (must have ".normalX16.pkl", or ".normalX32.pkl" suffix)
        normals: (3, H, W) xyz-vector
        mask: (1, H, W) binary mask for objects/background (1 = foreground, 0 = background)
        meta: [optional] a dictionary with metadata
    """
    if mask is None:
        mask = np.ones((1, normals.shape[0], normals.shape[1]), dtype=np.float32)

    xy_norm = normals[:2, :, :]
    z_norm = normals[2:3, :, :]

    suf = Path(path).suffixes[0]
    if suf == '.normalX16':
        write_flowou_X16(path, xy_norm, z_norm, mask, meta)
    elif suf == '.normalX32':
        write_flowou_X32(path, xy_norm, z_norm, mask, meta)
    else:
        raise ValueError(f"Incorrect normal path suffix: {Path(path).suffixes}")


def read_normals(path, return_meta=False):
    """Read png-compressed compressed png normals and mask
    Function uses existing infrastructure of write_flow* functions for saving normals

    Args:
        path: read path (must have ".normalX16.pkl", or ".normalX32.pkl" suffix)
        return_meta: [optional] return metadata

    Returns:
        normals: (3, H, W) xyz-vector
        mask: (1, H, W) binary mask for objects/background (1 = foreground, 0 = background)
        meta: [optional] a dictionary with metadata
    """
    suf = Path(path).suffixes[0]
    if suf == '.normalX16':
        data = read_flowou_X16(path, return_meta)
    elif suf == '.normalX32':
        data = read_flowou_X32(path, return_meta)
    else:
        raise ValueError(f"Incorrect normal path suffix: {Path(path).suffixes}")

    xy_norm, z_norm, mask = data[0], data[1], data[2]
    normals = np.concatenate([xy_norm, z_norm], axis=0)
    if return_meta:
        metadata = data[3]
        return normals, mask, metadata
    else:
        return normals, mask

def wuff_decode(buffer):
    """Faster decoding of PNGs, but getting Illegal hardware instruction coredump on Radon"""
    from pywuffs import ImageDecoderType, PixelFormat
    from pywuffs.aux import (
        ImageDecoder,
        ImageDecoderConfig,
        # ImageDecoderFlags
    )
    config = ImageDecoderConfig()

    # All decoders are enabled by default
    config.enabled_decoders = [ImageDecoderType.PNG]

    # No metadata is reported by default
    # config.flags = [ImageDecoderFlags.REPORT_METADATA_EXIF]

    # Pixel format is PixelFormat.BGRA_PREMUL by default
    config.pixel_format = PixelFormat.BGR

    decoder = ImageDecoder(config)
    decoding_result = decoder.decode(buffer.tobytes())

    # Decoded image data in BGR format
    image_data = decoding_result.pixbuf
    return image_data

@profile
def read_flowou_X16(path, return_meta=False, overwrite2lz4=True):
    @profile
    def decode_channel(data, encoding):
        buf = data['data']
        if encoding == 'lz4':
            xs_u16 = np.frombuffer(lz4.frame.decompress(buf), dtype=np.uint16)
            xs_compressed = einops.rearrange(xs_u16, '(H W) -> H W', H=data['height'], W=data['width'])
        elif encoding == 'png':
            ## Faster decoding of PNGs, but getting Illegal hardware instruction coredump on Radon {{{
            # try:
            #     xs_3u8 = wuff_decode(buf)
            # except ImportError:
            #     warnings.warn("PyWuffs (fast PNG decoder) could not be imported, falling back to OpenCV")
            # }}}
            xs_3u8 = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            xs_compressed = data_3u8_to_u16(xs_3u8)
        else:
            raise NotImplementedError(f"Encoding {encoding} not implemented")
        xs = decompress_channel(xs_compressed, data['min'], data['max'])
        return xs

    @profile
    def data_3u8_to_u16(xs):
        assert xs.dtype == np.uint8
        byte_3, byte_2, byte_1 = np.dsplit(np.uint16(xs), 3)

        u16 = (byte_2 << 8) | byte_1
        return einops.rearrange(u16, 'H W 1 -> H W')

    @profile
    def decompress_channel(compressed_xs, lb, ub):
        if not np.all(np.isfinite((lb, ub))):
            raise ValueError(f"FlowOU compression lower bound ({lb}) or upper bound ({ub}) invalid.")
        xs_01 = np.float32(compressed_xs) / (2**16 - 1)
        xs = (xs_01 * (ub - lb)) + lb
        if not np.all(np.isfinite(xs)):
            raise ValueError("FlowOU contains NaNs or infs.")
        return xs

    with open(path, 'rb') as fin:
        data = pickle.load(fin)

    encoding = data['encoding'] if 'encoding' in data else 'png'
    assert encoding == 'png' or encoding == 'lz4'

    flow_x = decode_channel(data['flow_x'], encoding)
    flow_y = decode_channel(data['flow_y'], encoding)
    flow = np.stack((flow_x, flow_y), axis=0)
    uncertainty = einops.rearrange(decode_channel(data['sigma'], encoding), 'H W -> 1 H W')
    occlusions = einops.rearrange(decode_channel(data['occlusion'], encoding), 'H W -> 1 H W')

    if overwrite2lz4 and encoding != 'lz4':
        write_flowou_X16(path, flow, occlusions, uncertainty, encoding='lz4')

    if return_meta:
        meta = data.get('meta', {})
        return flow, occlusions, uncertainty, meta
    else:
        return flow, occlusions, uncertainty


def compress_flowou(flow, occlusions, uncertainty,
                    write_fn=write_flowou_X16,
                    read_fn=read_flowou_X16):
    # create a temporary file in /dev/shm which is a RAMdisk on linux
    tmp_file = NamedTemporaryFile(delete=False, suffix=".flowouX16.pkl", dir="/dev/shm")
    tmp_file.close()

    write_fn(tmp_file.name, flow, occlusions, uncertainty)
    c_flow, c_occlusions, c_uncertainty = read_fn(tmp_file.name)

    os.unlink(tmp_file.name)
    return c_flow, c_occlusions, c_uncertainty


class GeneralVideoCapture(object):
    """A cv2.VideoCapture replacement, that can also read images in a directory

    args:
        path: a video path, or a image directory path, or a list of image Paths
    """

    def __init__(self, path, reverse=False, glob=None):
        self.i = 0
        if isinstance(path, list):
            self.image_inputs = True
            self.images = path
            self.path = None
            if reverse:
                self.images = self.images[::-1]
        elif Path(path).is_dir():
            self.image_inputs = True
            self.path = path
            if glob is None:
                self.images = sorted([f for f in next(os.walk(path))[2]
                                      if os.path.splitext(f)[1].lower() in ['.jpg', '.png', '.jpeg']])
            else:
                self.images = [str(x) for x in sorted(list(Path(path).glob(glob)))]

            if reverse:
                self.images = self.images[::-1]
        else:
            self.image_inputs = False
            self.cap = cv2.VideoCapture(str(path))

    def read(self):
        if self.image_inputs:
            if self.i >= len(self.images):
                return False, None
            img_path = str(self.images[self.i])
            if self.path is not None:
                img_path = os.path.join(self.path, img_path)
            self.frame_src = self.images[self.i]
            img = cv2.imread(img_path)
            self.i += 1
            return True, img
        else:
            self.i += 1
            return self.cap.read()

    def last_name(self):
        last_i = self.i - 1
        if self.image_inputs:
            # return os.path.splitext(self.images[last_i])[0]
            return Path(self.images[last_i]).stem
        else:
            return f'{last_i:08d}'

    def release(self):
        if self.image_inputs:
            return None
        else:
            return self.cap.release()


def get_video_frames(path, glob=None):
    """
    args:
        glob: something like "rgba_*.png"
    """
    cap = GeneralVideoCapture(path, glob=glob)
    while True:
        success, frame = cap.read()
        if not success or frame is None:
            return None
        yield frame

def get_named_video_frames(path):
    cap = GeneralVideoCapture(path)
    while True:
        success, frame = cap.read()
        if not success or frame is None:
            return None
        yield frame, cap.last_name()


def get_video_length(path, glob=None):
    N = 0
    for frame in get_video_frames(path, glob=glob):
        N += 1
    return N

def _flow_cache_preload_worker(input_queue, output_queue):
    while True:
        try:
            cache_path = input_queue.get(timeout=0)
        except queue.Empty:
            continue

        if cache_path is None:
            break

        output_queue.put((cache_path, read_flowou(cache_path)))


class FlowCache():
    def __init__(self, cache_dir, max_RAM_MB=10000, max_GPU_RAM_MB=5000,
                 preload=False, num_workers=0):
        """
        args:
            preload: [optional] preload flows for next `preload` frames in parallel (using multiprocessing)
            num_workers: [optional] Number of processes doing the pre-loading. Must be > 0 when using `preload`.
        """
        self.cache_dir = cache_dir
        self.max_RAM_MB = max_RAM_MB
        self.max_GPU_RAM_MB = max_GPU_RAM_MB
        self.ram_cache = {}
        self.gpu_ram_cache = {}
        self.bytes_used = 0
        self.gpu_ram_bytes_used = 0
        self.n_saved = 0
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # stuff related to multiprocessing cache pre-loading
        # the PNG decompression of the cached flows takes a very long time for big images, like 1920x1800 in CroHD
        # here we spawn several workers that get paths to cached flowou files in their input queues load the flowous,
        # and store them into a shared output_queue.
        # from there, the read_flowou_from_disk function either directly returns them or store them into a
        # preload_cache dictionary (cache_path: data) for later use
        # the preload_currently_queued set takes care not to preload a cached flow multiple times.
        # When reading th (left_id -> rigth_id) cache, we enqueue (left_id + 1 -> right_id + 1), ... up to
        # (left_id + preload - 1 -> right_id + preload - 1)
        # with preload=4, num_workers=15 we get about 4x speedup on CroHD. Not great, not terrible.
        # based on https://teddykoker.com/2020/12/dataloader/
        self.preload = preload
        if preload:
            self.preload_input_queues = []
            self.preload_output_queue = multiprocessing.Queue()
            self.preload_workers = []
            self.preload_cache = {}
            self.preload_currently_queued = set()

            assert num_workers > 0
            for _ in range(num_workers):
                input_queue = multiprocessing.Queue()
                worker = multiprocessing.Process(
                    target=_flow_cache_preload_worker, args=(input_queue, self.preload_output_queue))
                worker.daemon = True
                worker.start()
                self.preload_workers.append(worker)
                self.preload_input_queues.append(input_queue)
            logger.debug(f"Started {len(self.preload_workers)} preload workers")

            self.preload_worker_cycle = cycle(range(num_workers))

    def _put(self, key, value):
        for tensor in value:
            self.bytes_used += sys.getsizeof(tensor.untyped_storage())
        self.ram_cache[key] = value

    def _put_gpu_ram(self, key, value):
        for tensor in value:
            self.gpu_ram_bytes_used += sys.getsizeof(tensor.untyped_storage())
        self.gpu_ram_cache[key] = value

    def _get(self, key):
        return self.ram_cache[key]

    def _get_gpu_ram(self, key):
        return self.gpu_ram_cache[key]

    def ram_space_left(self):
        max_bytes = self.max_RAM_MB * 1000000
        return max(max_bytes - self.bytes_used, 0)

    def gpu_ram_space_left(self):
        max_bytes = self.max_GPU_RAM_MB * 1000000
        return max(max_bytes - self.gpu_ram_bytes_used, 0)

    def read_flowou_from_disk(self, cache_path):
        if not self.preload:
            return read_flowou(cache_path)
        else:
            # print(f"reading {cache_path}")
            # print(f"preload cache size: {len(self.preload_cache)}")
            # print(f"currently have: {', '.join(sorted([key.stem.split('.')[0] for key in self.preload_cache.keys()]))}")
            if cache_path in self.preload_cache:
                result = self.preload_cache[cache_path]
                del self.preload_cache[cache_path]
                self.preload_currently_queued.remove(cache_path)
                return result
            else:
                while True:
                    try:
                        preloaded_path, preloaded_data = self.preload_output_queue.get(timeout=0)
                    except queue.Empty:
                        continue  # keep waiting
                    if preloaded_path == cache_path:
                        self.preload_currently_queued.remove(cache_path)
                        return preloaded_data
                    else:  # cache for later
                        self.preload_cache[preloaded_path] = preloaded_data

    def preload_path(self, cache_path):
        if cache_path not in self.preload_currently_queued:
            self.preload_input_queues[next(self.preload_worker_cycle)].put(cache_path)
            self.preload_currently_queued.add(cache_path)

    def __del__(self):
        if self.preload:
            try:
                # ask the workers to quit
                for i, worker in enumerate(self.preload_workers):
                    logger.debug("Asking preload worker to stop")
                    self.preload_input_queues[i].put(None)
                    worker.join(timeout=5.0)
                for q in self.preload_input_queues:
                    q.cancel_join_thread()
                    q.close()
                self.preload_output_queue.cancel_join_thread()
                self.preload_output_queue.close()
            finally:
                # kill it if it still lives
                for w in self.preload_workers:
                    if w.is_alive():
                        logger.debug("Killing preload worker that refused to stop")
                        w.terminate()

    @profile
    def read(self, left_id, right_id, flower, rotation=0):
        flower_name = flower.C.name
        key = (flower_name, left_id, right_id, rotation)
        flow_left_to_right, occlusions, sigmas = None, None, None
        if key in self.gpu_ram_cache:
            flow_left_to_right, occlusions, sigmas = self._get_gpu_ram(key)
        elif key in self.ram_cache:
            flow_left_to_right, occlusions, sigmas = self._get(key)
            flow_left_to_right = flow_left_to_right.to('cuda')
            occlusions = occlusions.to('cuda')
            sigmas = sigmas.to('cuda')
        else:
            try:
                cache_dir = self.cache_dir / flower_name
                if rotation == 0:
                    cache_path = cache_dir / f'{left_id}--{right_id}.flowouX16.pkl'
                else:
                    cache_path = cache_dir / f'{left_id}--{right_id}_rot{rotation}.flowouX16.pkl'
                assert cache_path.exists()
                if self.preload:
                    assert rotation == 0, NotImplementedError('Preload is not implemented for rotation != 0')
                    for i in range(self.preload):
                        time_direction = 1 if right_id > left_id else -1
                        pre_cache_path = cache_dir / f'{left_id + i * time_direction}--{right_id + i * time_direction}.flowouX16.pkl'
                        if not pre_cache_path.exists():
                            # print(f"{pre_cache_path} not cached yet")
                            continue
                        self.preload_path(pre_cache_path)

                flow_left_to_right, occlusions, sigmas = self.read_flowou_from_disk(cache_path)

                flow_left_to_right = torch.from_numpy(flow_left_to_right).to('cuda')
                occlusions = torch.from_numpy(occlusions).to('cuda')
                sigmas = torch.from_numpy(sigmas).to('cuda')
                # when reading from disk, try to cache to GPU / RAM
                self.write(left_id, right_id, flow_left_to_right, occlusions, sigmas, flower, rotation=rotation)
            except Exception:
                pass

        return flow_left_to_right, occlusions, sigmas

    # @profile
    def write(self, left_id, right_id, flow_left_to_right, occlusions, sigmas, flower, rotation=0):
        flower_name = flower.C.name
        key = (flower_name, left_id, right_id, rotation)
        if self.gpu_ram_space_left() > 0:
            self._put_gpu_ram(key, (flow_left_to_right, occlusions, sigmas))
        elif self.ram_space_left() > 0:
            self._put(key, (flow_left_to_right.cpu(),
                            occlusions.cpu(),
                            sigmas.cpu()))
        else:
            cache_dir = self.cache_dir / flower_name
            cache_dir.mkdir(parents=True, exist_ok=True)
            if rotation == 0:
                cache_path = cache_dir / f'{left_id}--{right_id}.flowouX16.pkl'
            else:
                cache_path = cache_dir / f'{left_id}--{right_id}_rot{rotation}.flowouX16.pkl'
            if not cache_path.exists():
                write_flowou(cache_path,
                             ensure_numpy(flow_left_to_right),
                             ensure_numpy(occlusions),
                             ensure_numpy(sigmas))
        self.n_saved += 1

    def clear(self, clear_disk=False, only_used_methods=True):
        logger.debug(f'Saved {self.n_saved} flows, '
                     f'{len(self.gpu_ram_cache)} on GPU ({self.gpu_ram_bytes_used / 2**30:.2f}GiB), '
                     f'{len(self.ram_cache)} on RAM ({self.bytes_used / 2**30:.2f}GiB)')
        used_flow_names = self._all_flow_names()
        c = Counter()
        for flow_name, left_id, right_id, rotation in self.ram_cache.keys():
            delta = abs(left_id - right_id)
            if rotation != 0:
                delta = f'{delta}_{rotation}'
            c[delta] += 1
        logger.debug(f'delta frequency: {c}')

        self.gpu_ram_cache.clear()
        self.gpu_ram_bytes_used = 0
        self.ram_cache.clear()
        self.bytes_used = 0
        self.n_saved = 0

        if clear_disk:
            if only_used_methods:
                for flow_name in used_flow_names:
                    shutil.rmtree(self.cache_dir / flow_name, ignore_errors=True)
            else:
                shutil.rmtree(self.cache_dir, ignore_errors=True)

    def _all_flow_names(self):
        return set.union(
            set(flow_name for flow_name, _, _, _ in self.ram_cache.keys()),
            set(flow_name for flow_name, _, _, _ in self.gpu_ram_cache.keys()))

    def backup_to_disk(self):
        """Save all the cached flowous to disk"""
        n_saved = 0
        for (flow_name, left_id, right_id, rotation), val in self.ram_cache.items():
            cache_dir = self.cache_dir / flow_name
            cache_dir.mkdir(parents=True, exist_ok=True)
            if rotation == 0:
                cache_path = cache_dir / f'{left_id}--{right_id}.flowouX16.pkl'
            else:
                cache_path = cache_dir / f'{left_id}--{right_id}_rot{rotation}.flowouX16.pkl'
            if not cache_path.exists():
                write_flowou(cache_path, *[ensure_numpy(x) for x in val])
                n_saved += 1

        for (flow_name, left_id, right_id, rotation), val in self.gpu_ram_cache.items():
            cache_dir = self.cache_dir / flow_name
            cache_dir.mkdir(parents=True, exist_ok=True)
            if rotation == 0:
                cache_path = cache_dir / f'{left_id}--{right_id}.flowouX16.pkl'
            else:
                cache_path = cache_dir / f'{left_id}--{right_id}_rot{rotation}.flowouX16.pkl'
            if not cache_path.exists():
                write_flowou(cache_path, *[ensure_numpy(x) for x in val])
                n_saved += 1
        if n_saved > 0:
            logger.info(f"Saved {n_saved} cached flowous to disk.")

    def load_from_disk(self):
        raise NotImplementedError("not implemented. todo: load from all/some flow_name subdirectories")
        all_cached = sorted(list(self.cache_dir.glob('*.flowouX16.pkl')))
        n_loaded = 0
        for path in tqdm.tqdm(all_cached, desc="loading flowous from disk"):
            rotation = 0
            left_id, right_id = Path(path.stem).stem.split('--')
            if '_rot' in right_id:
                right_id, rot_str = right_id.split('_rot')
                rotation = int(rot_str)
            left_id, right_id = int(left_id), int(right_id)

            try:
                flow_left_to_right, occlusions, sigmas = read_flowou(path)
                flow_left_to_right = torch.from_numpy(flow_left_to_right).to('cuda')
                occlusions = torch.from_numpy(occlusions).to('cuda')
                sigmas = torch.from_numpy(sigmas).to('cuda')
                self.write(left_id, right_id, flow_left_to_right, occlusions, sigmas, rotation=rotation)
                n_loaded += 1
            except Exception:
                pass
        logger.info(f"Loaded {n_loaded} flowous into cache.")

    @classmethod
    def create(cls, base_dir, dataset_name, seq_name, cpu_limit_GB, gpu_limit_GB,
               persistent=False, preload_from_disk=False):
        cache_dir = base_dir / dataset_name / seq_name
        if not persistent:
            try:
                shutil.rmtree(cache_dir)
            except Exception:
                pass
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache = cls(cache_dir, max_RAM_MB=cpu_limit_GB * 1e3, max_GPU_RAM_MB=gpu_limit_GB * 1e3) 
        if persistent and preload_from_disk:
            cache.load_from_disk()

        return cache


# @profile
def get_flowou_with_cache(flower, left_img, right_img, flow_init=None,
                          cache=None, left_id=None, right_id=None,
                          read_cache=False, write_cache=False,
                          flow_timer=None, rotation=0):
    """Compute flow from left_img to right_img. Possibly with caching.

    args:
        flower: flow wrapper
        left_img: (H, W, 3) BGR np.uint8 image
        right_img: (H, W, 3) BGR np.uint8 image
        flow_init: [optional] (2, H, W) tensor with flow initialisation (caching is disabled when flow_init used)
        cache: [optional] cache object with
        left_id: [optional] frame number of left_img
        right_id: [optional] frame number of right_img
        read_cache: [optional] enable loading from flow cache
        write_cache: [optional] enable writing into flow cache

    returns:
        flowou: FlowOUTrackingResult
    """
    # print(f"Getting flow {left_id} -> {right_id}")
    # here to avoid circular import when MFT.results import io for the flowou read/write functions
    from MFTIQ.results import FlowOUTrackingResult
    if read_cache and flow_init is None:
        # attempt loading cached flow
        assert left_id is not None
        assert right_id is not None

        try:
            assert cache is not None
            flow_left_to_right, occlusions, sigmas = cache.read(left_id, right_id, flower, rotation=rotation)
            assert flow_left_to_right is not None
            flowou = FlowOUTrackingResult(flow_left_to_right, occlusions, sigmas)
            must_compute = False
        except Exception:
            must_compute = True
    else:
        must_compute = True

    if must_compute:  # read_cache == False, flow not cached yet, or some cache read error
        # print(f'computing flow {left_id}->{right_id}')
        if flow_timer is not None:
            flow_timer.start()
            # print(f'{left_img.shape=}')

        rotpad = RotateAndPad(degrees=rotation)
        left_img_rp, right_img_rp = rotpad(left_img, right_img)
        if flow_init is not None and rotation != 0:
            raise NotImplementedError('rotation is not implemented with flow init (yet)')
        flow_left_to_right, extra = flower.compute_flow(left_img_rp, right_img_rp, mode='flow',
                                                        init_flow=flow_init)
        flow_left_to_right, extra = rotpad.correct_flow(flow_left_to_right, extra)
        if flow_timer is not None:
            flow_timer.stop()
        flowou = FlowOUTrackingResult(flow_left_to_right, extra['occlusion'], extra['sigma'])
        occlusions, sigmas = extra['occlusion'], extra['sigma']

    if (cache is not None) and write_cache and must_compute and (flow_init is None):
        cache.write(left_id, right_id, flowou.flow, flowou.occlusion, flowou.sigma, flower, rotation=rotation)
    return flowou

class FeatureCache(object):
    def __init__(self, capacity, path=None):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.path = path

    def read(self, frame_i, model_name, rotation=0):
        key = f'{model_name}__{frame_i}__rot{rotation}'
        if key not in self.cache:
            # try to find it on disk
            if self.path is not None:
                if rotation == 0:
                    path = self.path / model_name / f'{frame_i:06d}.pkl'
                else:
                    path = self.path / model_name / f'{frame_i:06d}_rot{rotation}.pkl'
                if path.exists():
                    try:
                        with path.open('rb') as fin:
                            value = self._to(pickle.load(fin), 'cuda')
                        # print(f'Reading UOM features on frame #{frame_i} from disk')
                        # print(self.cache.keys())
                        # print(sorted([int(x.split('__')[1]) for x in self.cache.keys()]))
                        return value
                    except Exception:
                        logger.exception("Could not load disk-cached features.")
                            
            # print(f'Features on frame #{frame_i} not cached')
            return {}
        else:
            # print(f'Features on frame #{frame_i} ARE cached :D')
            self.cache.move_to_end(key)
            return self.cache[key]

    def write(self, frame_i, value, model_name, rotation=0):
        key = f'{model_name}__{frame_i}__rot{rotation}'
        if key not in self.cache:
            self.cache[key] = value
        self.cache.move_to_end(key)

        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

        if self.path is not None:
            if rotation == 0:
                path = self.path / model_name / f'{frame_i:06d}.pkl'
            else:
                path = self.path / model_name / f'{frame_i:06d}_rot{rotation}.pkl'
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)

                cpu_value = self._to(value, 'cpu')
                with path.open('wb') as fout:
                    pickle.dump(cpu_value, fout)

    def _to(self, value, device):
        """
        args:
            value: a dict containing torch tensors
        """
        result = {}
        for k, v in value.items():
            result[k] = v.to(device)

        return result

def slugify_path(path):
    return ''.join(c if c.isalnum() else '-' for c in str(path))
