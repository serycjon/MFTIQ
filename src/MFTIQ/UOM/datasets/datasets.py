# -*- origami-fold-style: triple-braces -*-
# Data loading based on https://github.com/NVIDIA/flownet2-pytorch
# MFTIQ - WACV2025
import copy

import numpy as np
import torch
import torch.utils.data as data
import logging

import os
import sys
import random
from glob import glob
import os.path as osp

from MFTIQ.RAFT.core.utils import frame_utils
from MFTIQ.RAFT.core.utils.augmentor import FlowAugmenter, SparseFlowAugmenter
from MFTIQ.UOM.datasets.augmentor_dict import FlowAugmenterDict

# from MFTIQ.RAFT.multiflow_from_kubric import get_multiflow
from MFTIQ.UOM.create_kubric_data import load_data_creation_config, CompStatus

from MFTIQ.UOM.create_kubric_data import get_multiflow
from MFTIQ.UOM.create_kubric_data import compute_occl_valid

from copy import deepcopy
from MFTIQ.utils.io import read_flowou

import pickle as pk
from pathlib import Path
import einops
import cv2
import datetime
from ipdb import iex

import h5py
import time

from rich.console import Console
console = Console()

# flow is encoded with sign, so 2**15, occlusion and uncertainty without sign, so 2**16:
FLOWOU_IO_FLOW_MULTIPLIER = 2**5  # max-abs-val = 2**(15-5) = 1024, step = 2**(-5) = 0.03
FLOWOU_IO_OCCLUSION_MULTIPLIER = 2**15  # max-val = 2**(16-15) = 2, step = 2**(-15) = 3e-5
FLOWOU_IO_UNCERTAINTY_MULTIPLIER = 2**9   # max-val = 2**(16-9) = 128, step = 2**(-9) = 0.0019


def standardize_outputs(img1=None, img2=None, flow=None, valid=None, occl=None, **kwargs):
    output_data = {'img1': img1,
                   'img2': img2,
                   'flow': flow,
                   'valid': valid,
                   'occl': occl}

    if not kwargs.get('extended_dataset', False):
        return output_data

    output_data['flow_est'] = kwargs.get('flow_est', flow.clone() + torch.randn_like(flow))
    if kwargs.get('extended_dataset') == 'flow_est':
        return output_data

    return output_data


def read_flowou_png(path):
    """Read png-compressed flow, occlusions and uncertainty

    Args:
        path: ".flowou.png" file path

    Returns:
        flow: (2, H, W) float32 numpy array (delta-x, delta-y)
        occlusions: (1, H, W) float32 array with occlusion scores (1 = occlusion, 0 = visible)
        uncertainty: (1, H, W) float32 array with uncertainty sigma (0 = dirac)
    """
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


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, load_occlusion=False, root=None, **kwargs):
        self.root = root
        self.augmentor = None
        self.sparse = sparse
        self.load_occlusion = load_occlusion
        self.extended_dataset = kwargs.get('extended_dataset', False)

        if kwargs.get('augmentor_for_dict', False) and aug_params is not None:
            self.augmentor = FlowAugmenterDict(**aug_params)
        elif aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmenter(**aug_params, load_occlusion=load_occlusion)
            else:
                self.augmentor = FlowAugmenter(**aug_params, load_occlusion=load_occlusion)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.occlusion_list = []
        self.image_list = []
        self.extra_info = []
        self.num_repetitions = 1

        self.multi_flow_list = []
        self.multi_image_list = []
        self.multi_occl_list = []
        self.flow_zero_list = []
        self.flow_est_list = []

        self.logger = logging.getLogger(f'{self.__class__.__name__}')

    def get_reference_frame_path(self, index, relative=False):
        cpath = self.image_list[index][0]
        if relative:
            cpath = cpath.replace(self.root, '')
        return cpath

    def normalise_occlusions_01(self, occl):
        if occl.max() >= 1.1:
            return occl / 255.0
        else:
            return occl

    @iex
    def __getitem__(self, index):
        """
        returns:
            img1: (3, H, W) float32 tensor with 0-255 RGB(!) values
            img2: (3, H, W) float32 tensor with 0-255 RGB(!) values
            flow: (2, H, W) float32 tensor with (xy-ordered?) flow
            valid: (1, H, W) float32 tensor with values 0 (invalid), and 1 (valid)
            occl: (1, H, W) float32 tensor with 0-1 occlusion mask
        """
        index = index % len(self.image_list)

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = einops.rearrange(torch.from_numpy(img1), 'H W C -> C H W', C=2).float()
            img2 = einops.rearrange(torch.from_numpy(img2), 'H W C -> C H W', C=2).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(np.random.randint(0, 1024) + worker_info.id)
                np.random.seed(np.random.randint(0, 1024) + worker_info.id)
                random.seed(np.random.randint(0, 1024) + worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.read_gen_sparse_flow(self.flow_list[index])
            valid = einops.rearrange(valid, 'H W -> H W 1')  # np.expand_dims(valid, 2)
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = einops.repeat(img1, 'H W -> H W C', C=3)
            img2 = einops.repeat(img2, 'H W -> H W C', C=3)
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.load_occlusion:
            occl = frame_utils.read_gen(self.occlusion_list[index])
            occl = np.array(occl).astype(np.float32)
            occl = self.normalise_occlusions_01(occl)
        else:
            H, W, C = img1.shape
            occl = np.zeros([H, W, 1], dtype=np.float32)

        if len(occl.shape) == 2:
            occl = einops.rearrange(occl, 'H W -> H W 1')  # occl = np.expand_dims(occl, axis=2)
        else:
            occl = occl[:, :, 0:1]

        if self.augmentor is not None:
            # if self.sparse:
            #     img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            # else:
            #     img1, img2, flow = self.augmentor(img1, img2, flow)
            # orig_occl = occl.copy()
            img1, img2, flow, valid, occl = self.augmentor(img1, img2, flow, valid, occl)

        img1 = einops.rearrange(torch.from_numpy(img1), 'H W C -> C H W', C=3).float()
        img2 = einops.rearrange(torch.from_numpy(img2), 'H W C -> C H W', C=3).float()
        flow = einops.rearrange(torch.from_numpy(flow), 'H W xy -> xy H W', xy=2).float()
        occl = einops.rearrange(torch.from_numpy(occl), 'H W 1 -> 1 H W').float()

        if valid is not None:
            valid = einops.rearrange(torch.from_numpy(valid), 'H W 1 -> 1 H W') > 0.99
            valid = valid & einops.rearrange(torch.all(flow.abs() < 1000, dim=0), 'H W -> 1 H W')
        else:
            valid = einops.rearrange(torch.all(flow.abs() < 1000, dim=0), 'H W -> 1 H W')

        return standardize_outputs(img1, img2, flow, valid.float(), occl, extended_dataset=self.extended_dataset)

    def __rmul__(self, v):
        assert isinstance(v, int)
        self.num_repetitions *= v
        return self

    def __len__(self):
        return len(self.image_list) * self.num_repetitions

    def load_cache(self, file_path):
        self.logger.info("Loading cache")
        file_path = f'{file_path}.pkl'
        if not os.path.isfile(file_path):
            return False
        with open(file_path, 'rb') as f:
            # files = np.load(f, allow_pickle=True)
            files = pk.load(f)

        self.image_list = files.get('image_list')
        self.flow_list = files.get('flow_list')
        self.occlusion_list = files.get('occlusion_list', [])
        self.multi_flow_list = files.get('multi_flow_list', [])
        self.multi_image_list = files.get('multi_image_list', [])
        self.multi_occl_list = files.get('multi_occl_list', [])
        self.extra_info = files.get('extra_info', [])
        self.flow_zero_list = files.get('flow_zero_list', [])

        self.flow_est_list = files.get('flow_est_list', [])

        self.logger.info("Done loading cache")
        return True

    def save_cache(self, file_path, additional_files=None):
        file_path = f'{file_path}.pkl'
        files = {'image_list': self.image_list,
                 'flow_list': self.flow_list,
                 'occlusion_list': self.occlusion_list,
                 'extra_info': self.extra_info}
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if additional_files is not None:
            files.update(additional_files)
        with open(file_path, 'wb') as f:
            pk.dump(files, f)
            # np.save(f, files, allow_pickle=True)

    @staticmethod
    def bw_bilinear_interpolate_flow_numpy(im, flow):

        def _bw_bilin_interp(im, x, y):
            x = np.asarray(x)
            y = np.asarray(y)

            x0 = np.floor(x).astype(int)
            x1 = x0 + 1
            y0 = np.floor(y).astype(int)
            y1 = y0 + 1

            x0 = np.clip(x0, 0, im.shape[1] - 1)
            x1 = np.clip(x1, 0, im.shape[1] - 1)
            y0 = np.clip(y0, 0, im.shape[0] - 1)
            y1 = np.clip(y1, 0, im.shape[0] - 1)

            Ia = im[y0, x0]
            Ib = im[y1, x0]
            Ic = im[y0, x1]
            Id = im[y1, x1]

            wa = (x1 - x) * (y1 - y)
            wb = (x1 - x) * (y - y0)
            wc = (x - x0) * (y1 - y)
            wd = (x - x0) * (y - y0)

            return wa * Ia + wb * Ib + wc * Ic + wd * Id

        ndim = im.ndim
        if ndim == 2:
            im = np.expand_dims(im, axis=2)
        H, W, C = im.shape
        X_g, Y_g = np.meshgrid(range(W), range(H))
        x, y = flow[:, :, 0], flow[:, :, 1]
        x = x + X_g
        y = y + Y_g
        im_w = []
        for i in range(C):
            im_w.append(_bw_bilin_interp(im[:, :, i], x, y))
        im_w = np.stack(im_w, axis=2)

        if ndim == 2:
            im_w = im_w[:, :, 0]
        return im_w


class KubricLong2024Dataset(FlowDataset):
    def __init__(self, aug_params=None, split='train',
                 root=None,
                 load_occlusion=False,
                 upsample2=False, correct_flow=False, subsplit=None, mftiqdata=False, **kwargs):
        """
        """
        base_root_dirs = [Path('datasets/kubric_datasets/002_030'),
                          Path('datasets/kubric_datasets/030_150')]
        root = root if root is not None else base_root_dirs
        super().__init__(aug_params, load_occlusion=load_occlusion, root=root, augmentor_for_dict=True, **kwargs)

        self.flow_est_list = []
        self.upsample2 = upsample2
        self.correct_flow = correct_flow
        self.subsplit_fraction = 50 # each Xth data are set to be validation data, rest is the training data

        self.load_occl_valid_masks = kwargs.get('load_occl_valid_masks', True)

        assert split in ['train']
        if split == 'test':
            self.is_test = True

        if subsplit is not None:
            assert subsplit in ['train', 'validation', 'debug']
            self.save_file_path = f'train_files_lists/Kubric_Long_{split}_{subsplit}'
        else:
            self.save_file_path = f'train_files_lists/Kubric_Long_{split}'

        if isinstance(root, str):
            root = Path(root)
        if not isinstance(root, list):
            root = [root]

        if self.load_cache(self.save_file_path):
            print(f'Reading training data from CACHE {self.save_file_path}')
            return

        print(f'CACHE does not exist. Creating cache {self.save_file_path}')
        for data_root in root:
            for idx, scene_path in enumerate(sorted(data_root.iterdir())):
                if subsplit == 'train':
                    if idx % self.subsplit_fraction == 0:
                        continue # skip validation files
                elif subsplit == 'validation':
                    if idx % self.subsplit_fraction != 0:
                        continue # skip training files
                elif subsplit == 'debug':
                    if len(self.flow_list) > 4: # debugging
                        continue
                else:
                    raise NotImplementedError

                status = load_data_creation_config(scene_path)[-1]
                if status != CompStatus.CHECKED:
                    continue
                flow_gt_path = scene_path / 'flow_gt.flowouX16.pkl'
                if not flow_gt_path.is_file():
                    continue

                scene = scene_path.name

                self.image_list += [[scene_path/'rgb_reference.png', scene_path/'rgb_target.png']]
                self.flow_list += [scene_path/'flow_gt.flowouX16.pkl']
                self.flow_est_list += [scene_path / 'flow_est.flowouX16.pkl']

        additional_files = {'flow_est_list': self.flow_est_list}
        self.save_cache(self.save_file_path, additional_files=additional_files)

    def __getitem__(self, index):
        # print(index)
        min_fraction = 0.25
        data = self.__getitem_inner__(index)
        img_shape = einops.parse_shape(data['img1'], 'C H W')
        min_matched_area = min_fraction * img_shape['H'] * img_shape['W']

        # for _ in range(10):
        #     data = self.__getitem_inner__(index)
        #     matched_area = torch.sum(torch.logical_and(data['valid'] > 0.99, data['occl'] < 0.01))
        #     if matched_area >= min_matched_area:
        #         return data
        return data

    def __getitem_inner__(self, index):
        """
        returns:
            img1: (3, H, W) float32 tensor with 0-255 RGB(!) values
            img2: (3, H, W) float32 tensor with 0-255 RGB(!) values
            flow: (2, H, W) float32 tensor with (xy-ordered?) flow
            valid: (1, H, W) float32 tensor with values 0 (invalid), and 1 (valid)
            occl: (1, H, W) float32 tensor with 0-1 occlusion mask
        """
        index = index % len(self.image_list)

        if self.is_test:
            raise NotImplementedError
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = einops.rearrange(torch.from_numpy(img1), 'H W C -> C H W', C=2).float()
            img2 = einops.rearrange(torch.from_numpy(img2), 'H W C -> C H W', C=2).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(np.random.randint(0, 1024) + worker_info.id)
                np.random.seed(np.random.randint(0, 1024) + worker_info.id)
                random.seed(np.random.randint(0, 1024) + worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        data = {}

        # READ
        if self.load_occl_valid_masks:
            data['flow'], data['occl'], data['valid'] = read_flowou(self.flow_list[index])
        else:
            data['valid'] = None
            data['flow'], data['occl'], _ = read_flowou(self.flow_list[index])
        data['img1'] = frame_utils.read_gen(self.image_list[index][0])
        data['img2'] = frame_utils.read_gen(self.image_list[index][1])

        assert len(self.flow_list) == len(self.flow_est_list)

        flow_pick_prob = np.random.uniform(0.0, 1.0)
        data['flow_est'], _, _ = read_flowou(self.flow_est_list[index])

        # NORMALISE SHAPES (FOR AUGMENTOR)
        for k, v in data.items():
            if 'flow' in k:
                data[k] = einops.rearrange(np.array(v).astype(np.float32), 'xy H W -> H W xy', xy=2)
            elif 'img' in k:
                data[k] = np.array(v).astype(np.uint8)
            elif 'occl' in k:
                data[k] = einops.rearrange(np.array(v).astype(np.float32), '1 H W -> H W 1')
            elif 'valid' in k and data['valid'] is not None:
                data[k] = einops.rearrange(np.array(v).astype(np.float32), '1 H W -> H W 1')

        if self.augmentor is not None:
            data = self.augmentor(data)

        for k, v in data.items():
            if 'img' in k:
                data[k] = einops.rearrange(torch.from_numpy(v), 'H W C -> C H W', C=3).float()
            elif 'flow' in k:
                data[k] = einops.rearrange(torch.from_numpy(v), 'H W xy -> xy H W', xy=2).float()
            elif 'occl' in k:
                data[k] = einops.rearrange(torch.from_numpy(v), 'H W 1 -> 1 H W').float()

        if 'valid' in data and data['valid'] is not None:
            data['valid'] = einops.rearrange(torch.from_numpy(data['valid']), 'H W 1 -> 1 H W') > 0.99
            data['valid'] = data['valid'] & einops.rearrange(torch.all(data['flow'].abs() < 1000, dim=0), 'H W -> 1 H W')
        else:
            data['valid'] = einops.rearrange(torch.all(data['flow'].abs() < 1000, dim=0), 'H W -> 1 H W')
        data['valid'] = data['valid'].float()
        return standardize_outputs(**data, extended_dataset=self.extended_dataset)


class KubricDataset(FlowDataset):
    def __init__(self, aug_params=None, split='train',
                 root='datasets/kubric_movi_e_longterm', load_occlusion=False,
                 upsample2=False, correct_flow=False, **kwargs):
        """
        """
        super(KubricDataset, self).__init__(aug_params, load_occlusion=load_occlusion, root=root, **kwargs)
        self.flow_zero_list = []
        self.multi_flow_list = []
        self.multi_image_list = []
        self.upsample2 = upsample2
        self.correct_flow = correct_flow

        if split == 'test':
            self.is_test = True

        self.save_file_path = f'train_files_lists/Kubric_Pixel_Tracking_{split}'

        if not self.load_cache(self.save_file_path):
            data_root = osp.join(root, split)

            for idx, scene in enumerate(os.listdir(data_root)):
                # if idx >= 9:
                #     break

                image_list = sorted(glob(osp.join(data_root, scene, 'images', '*.png')))
                flow_list = sorted(glob(osp.join(data_root, scene, 'flowou', '*.flowou.png')))

                for i in range(len(image_list) - 1):
                    self.image_list += [[image_list[0], image_list[i + 1]]]
                    self.extra_info += [(scene, i)]  # scene and frame_id

                    if split != 'test':
                        # +1 because of flow from 0 to 0 (first flow is saved only to see problems with discretisation)
                        self.flow_list += [flow_list[i+1]]
                        self.flow_zero_list += [flow_list[0]]

                self.multi_image_list.append(image_list)
                self.multi_flow_list.append(flow_list)

            self.save_cache(self.save_file_path, {'extra_info': self.extra_info,
                                                  'multi_image_list': self.multi_image_list,
                                                  'multi_flow_list': self.multi_flow_list,
                                                  'flow_zero_list': self.flow_zero_list})

    def get_data_delta(self, index, delta=None):
        if delta is None:
            im1_path = self.image_list[index][0]
            im2_path = self.image_list[index][1]
        else:
            im1_path = self.multi_image_list[index][0]
            im2_path = self.multi_image_list[index][delta]

        if self.is_test:
            img1 = frame_utils.read_gen(im1_path)
            img2 = frame_utils.read_gen(im2_path)
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if delta is None:
            flowou_path = self.flow_list[index]
            flowou_zero_path = self.flow_zero_list[index]
        else:
            flowou_path = self.multi_flow_list[index][delta]
            flowou_zero_path = self.multi_flow_list[index][0]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None

        flow, occlusions, _ = read_flowou_png(flowou_path)
        flow = einops.rearrange(flow, 'C H W -> H W C', C=2).astype(np.float32)
        occl = einops.rearrange(occlusions, 'C H W -> H W C', C=1).astype(np.float32)
        occl = self.normalise_occlusions_01(occl)

        if self.correct_flow and delta != 0:
            flow_zero, _, _ = read_flowou_png(flowou_zero_path)
            flow_zero = einops.rearrange(flow_zero, 'C H W -> H W C', C=2).astype(np.float32)
            obj_mask_bin = flow_zero[:, :, 0] > 0.25  # should be exactly 0.0 or 0.5, but there is error due to saving to discrete flowou  # noqa E501
            obj_mask_float = obj_mask_bin.astype(np.float32) - 0.5
            flow_zero[np.logical_not(obj_mask_bin)] = 0.
            flow_zero[obj_mask_bin] = 0.5

            flow = flow - flow_zero
            flow = self.bw_bilinear_interpolate_flow_numpy(flow, -flow_zero)
            obj_mask_float = self.bw_bilinear_interpolate_flow_numpy(obj_mask_float, -flow_zero) + 0.5
            occl = self.bw_bilinear_interpolate_flow_numpy(occl, -flow_zero)
            valid = np.logical_or(obj_mask_float > 0.99, obj_mask_float < 0.01)
            valid = np.expand_dims(valid, axis=2).astype(float)

        img1 = frame_utils.read_gen(im1_path)
        img2 = frame_utils.read_gen(im2_path)

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        if self.augmentor is not None:
            img1, img2, flow, valid, occl = self.augmentor(img1, img2, flow, valid, occl)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        occl = torch.from_numpy(occl).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid).permute(2, 0, 1).float() > 0.99
            valid = valid & torch.unsqueeze((flow[0].abs() < 1000) & (flow[1].abs() < 1000), dim=0)
        else:
            valid = torch.unsqueeze((flow[0].abs() < 1000) & (flow[1].abs() < 1000), dim=0)

        return standardize_outputs(img1, img2, flow, valid.float(), occl, extended_dataset=self.extended_dataset)

    def __getitem__(self, index):
        return self.get_data_delta(index)


class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/Sintel-complete',
                 dstype='clean', load_occlusion=False, subsplit=None, **kwargs):
        """
        :param subsplit: None : whole training dataset
                         'validation' : self.validation_subplit_dirs sequences only
                         'training' : all sequences except validation sequences
        """
        super(MpiSintel, self).__init__(aug_params, load_occlusion=load_occlusion, root=root, **kwargs)
        self.logger = logging.getLogger(f'{self.__class__.__name__}-{dstype}')

        if split == 'test':
            self.is_test = True
        self.validation_subsplit_dirs = ['alley_1', 'ambush_6', 'bamboo_2', 'cave_4', 'market_5', 'shaman_3']

        if subsplit is not None:
            self.save_file_path = f'train_files_lists/MpiSintel_{split}_{dstype}_{subsplit}'
        else:
            self.save_file_path = f'train_files_lists/MpiSintel_{split}_{dstype}'

        if not self.load_cache(self.save_file_path):
            self.logger.warning(f"Could not load cache from {self.save_file_path}")
            flow_root = osp.join(root, split, 'flow')
            occl_root = osp.join(root, split, 'occlusions_rev')
            image_root = osp.join(root, split, dstype)

            for scene in os.listdir(image_root):

                if subsplit is not None:
                    if subsplit == 'training' and scene in self.validation_subsplit_dirs:
                        continue
                    elif subsplit == 'validation' and scene not in self.validation_subsplit_dirs:
                        continue

                image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
                for i in range(len(image_list) - 1):
                    self.image_list += [[image_list[i], image_list[i + 1]]]
                    self.extra_info += [(scene, i)]  # scene and frame_id

                if split != 'test':
                    self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))
                    self.occlusion_list += sorted(glob(osp.join(occl_root, scene, '*.png')))

            self.save_cache(self.save_file_path)


class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='datasets/FlyingChairs_release/data', **kwargs):
        super(FlyingChairs, self).__init__(aug_params, root=root, **kwargs)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images) // 2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split == 'training' and xid == 1) or (split == 'validation' and xid == 2):
                self.flow_list += [flows[i]]
                self.image_list += [[images[2 * i], images[2 * i + 1]]]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/FlyingThings3D',
                 dstype='frames_cleanpass', load_occlusion=False, **kwargs):
        super(FlyingThings3D, self).__init__(aug_params, load_occlusion=load_occlusion, root=root, **kwargs)

        self.save_file_path = f'train_files_lists/FlyingThings3D_{dstype}'
        if not self.load_cache(self.save_file_path):
            for cam in ['left']:
                for direction in ['into_future', 'into_past']:
                    image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                    image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                    flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                    flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                    for idir, fdir in zip(image_dirs, flow_dirs):
                        images = sorted(glob(osp.join(idir, '*.png')))
                        flows = sorted(glob(osp.join(fdir, '*.pfm')))
                        for i in range(len(flows) - 1):
                            occl, im1, im2, flow = None, None, None, None
                            if direction == 'into_future':
                                im1 = images[i]
                                im2 = images[i+1]
                                flow = flows[i]
                                occl = flows[i].replace('optical_flow', 'optical_flow_occlusion_png') \
                                               .replace('.pfm', '.png')
                            elif direction == 'into_past':
                                im1 = images[i + 1]
                                im2 = images[i]
                                flow = flows[i + 1]
                                occl = flows[i + 1].replace('optical_flow', 'optical_flow_occlusion_png') \
                                                   .replace('.pfm', '.png')

                            if all([os.path.isfile(x) for x in [occl, im1, im2, flow]]):
                                self.image_list += [[im1, im2]]
                                self.flow_list += [flow]
                                self.occlusion_list += [occl]

            self.save_cache(self.save_file_path)


class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI/basic/', **kwargs):
        super(KITTI, self).__init__(aug_params, sparse=True, root=root, **kwargs)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))

        print(len(self.flow_list))


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1K', **kwargs):
        super(HD1K, self).__init__(aug_params, sparse=True, root=root, **kwargs)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows) - 1):
                self.flow_list += [flows[i]]
                self.image_list += [[images[i], images[i + 1]]]

            seq_ix += 1


class VIPER(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/PlayingForBenchmarks', **kwargs):
        super(VIPER, self).__init__(aug_params, sparse=True, root=root, **kwargs)

        if split == 'testing':
            self.is_test = True
            prefix = 'test'
        else:
            prefix = 'train'

        cur_dir = os.path.join(root, prefix, 'flow')
        subdirs = [d for d in os.listdir(cur_dir) if os.path.isdir(os.path.join(cur_dir, d))]
        subdirs.sort()

        split_list = np.loadtxt('viper_split.txt', dtype=np.int32)
        sl_idx = 0

        for sd in subdirs:
            for i in range(0, 10000, 10):

                flow_path = os.path.join(cur_dir, sd, '{:s}_{:05d}.npz'.format(sd, i))
                if os.path.isfile(flow_path):
                    image1_path = os.path.join(root, prefix, 'img', sd, '{:s}_{:05d}.jpg'.format(sd, i))
                    image2_path = os.path.join(root, prefix, 'img', sd, '{:s}_{:05d}.jpg'.format(sd, i+1))
                    frame_id = '{:s}_{:05d}.jpg'.format(sd, i)

                    if os.path.isfile(image1_path) and os.path.isfile(image2_path):
                        xid = split_list[sl_idx]
                        if (split == 'training' and xid == 1) or (split == 'validation' and xid == 2):
                            self.extra_info += [[frame_id]]
                            self.flow_list += [flow_path]
                            self.image_list += [[image1_path, image2_path]]
                        sl_idx += 1

        print('N of samples: ', len(self.flow_list))


class NoGoodAugmentationError(Exception):
    pass


def enough_matched(img, valid, occl, min_fraction):
    min_matched_area = min_fraction * img.shape[0] * img.shape[1]
    matched_area = np.sum(np.logical_and(valid > 0.99, occl < 0.01))
    return matched_area >= min_matched_area


class KubricLongflowDataset(data.Dataset):
    def __init__(self, aug_params=None, root='datasets/kubric_longtermflow_dataset/RES_1024x1024/', **kwargs):
        self.root = Path(root)
        self.augmentor = None
        self.extended_dataset = kwargs.get('extended_dataset', False)
        if aug_params is not None:
            self.augmentor = FlowAugmenter(**aug_params, load_occlusion=True)

        seed_dirs = sorted(list(self.root.glob('*')))
        self.seq_dirs = [x
                         for seed_dir in seed_dirs
                         for x in list(seed_dir.glob('*'))]
        self.seq_lengths = [len(list(seq_dir.glob('rgba_*'))) for seq_dir in self.seq_dirs]
        self.init_seed = False
        self.min_delta = 1
        self.max_delta = np.inf

    @iex
    def __getitem__(self, _):
        """
        returns:
            img1: (3, H, W) float32 tensor with 0-255 RGB(!) values
            img2: (3, H, W) float32 tensor with 0-255 RGB(!) values
            flow: (2, H, W) float32 tensor with (xy-ordered?) flow
            valid: (1, H, W) float32 tensor with values 0 (invalid), and 1 (valid)
            occl: (1, H, W) float32 tensor with 0-1 occlusion mask
        """
        for i_try in range(100):
            try:
                img1, img2, flow, valid, occl = self.__getitem_inner()
                # print(f'outer KUBRIC try count: {i_try + 1}')
                break
            except NoGoodAugmentationError:
                continue
        return img1, img2, flow, valid, occl

    def __getitem_inner(self):
        """
        returns:
            img1: (3, H, W) float32 tensor with 0-255 RGB(!) values
            img2: (3, H, W) float32 tensor with 0-255 RGB(!) values
            flow: (2, H, W) float32 tensor with (xy-ordered?) flow
            valid: (1, H, W) float32 tensor with values 0 (invalid), and 1 (valid)
            occl: (1, H, W) float32 tensor with 0-1 occlusion mask
        """
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        seq_i = np.random.choice(len(self.seq_dirs))
        seq_dir = self.seq_dirs[seq_i]
        seq_len = self.seq_lengths[seq_i]
        sampling_counter = 0
        while True:
            sampling_counter += 1
            left_frame_id, right_frame_id = np.random.choice(seq_len, size=2, replace=False)
            delta = np.abs(left_frame_id - right_frame_id)
            if delta >= self.min_delta and delta <= self.max_delta:
                break
            if sampling_counter > 1000:
                raise RuntimeError("cannot sample a suitable pair of frames")
        # step = 3
        # left_frame_id = np.random.choice(seq_len - step, size=1, replace=False)[0]
        # right_frame_id = left_frame_id + step
        multiflow_data = get_multiflow(seq_dir, frames=[left_frame_id, right_frame_id])
        flow = multiflow_data['flow'][1].astype(np.float32)  # flow into the right_frame (xy, H, W)
        flow = einops.rearrange(flow, 'xy H W -> H W xy', xy=2)
        # img1 = frame_utils.read_gen(seq_dir / f'rgba_{left_frame_id:05d}.png')
        img1 = multiflow_data['rgb'][0]  # (H, W, 3)
        img2 = multiflow_data['rgb'][1]

        H, W, C = img1.shape
        occl = multiflow_data['occlusion'][1].astype(np.float32)  # occlusion into the right_frame (1, H, W)
        occl = einops.rearrange(occl, '1 H W -> H W 1')
        valid = np.ones_like(occl)

        img1 = img1[..., :3]
        img2 = img2[..., :3]

        min_fraction = 0.25
        if not enough_matched(img1, valid, occl, min_fraction):
            raise NoGoodAugmentationError(":/")

        if self.augmentor is not None:
            # if self.sparse:
            #     img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            # else:
            #     img1, img2, flow = self.augmentor(img1, img2, flow)

            # flow shape (H, W, 2), valid shape (H, W, 1)?, occl shape (H, W, 1)
            good_augmentation_found = False
            for i_try in range(20):
                aug_img1, aug_img2, aug_flow, aug_valid, aug_occl = self.augmentor(img1, img2, flow, valid, occl)
                if enough_matched(aug_img1, aug_valid, aug_occl, min_fraction):
                    good_augmentation_found = True
                    # print(f'inner Kubric try count: {i_try + 1}')
                    break
            if not good_augmentation_found:
                # print(f'inner Kubric try count: {i_try + 1}')
                raise NoGoodAugmentationError(":/")  # try with other image pair
            else:
                img1, img2, flow, valid, occl = aug_img1, aug_img2, aug_flow, aug_valid, aug_occl

        if True:
            vis_path = Path('/tmp/kubric_trn_vis/')
            vis_path.mkdir(parents=True, exist_ok=True)
            # vis1 = cv2.cvtColor(cv2.cvtColor(img1[:, :, ::-1].copy(), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            # vis2 = cv2.cvtColor(cv2.cvtColor(img2[:, :, ::-1].copy(), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            vis1 = img1[:, :, ::-1].copy()
            vis2 = img2[:, :, ::-1].copy()
            H_vis, W_vis = vis1.shape[:2]
            vis1[occl[:, :, 0] > 0, :] = (0.2 * vis1[occl[:, :, 0] > 0, :]).astype(np.uint8)
            for i in range(25):
                color = tuple(np.random.randint(0, 255, size=3).astype(np.int32).tolist())
                found = False
                for i in range(300):  # 100 tries
                    y = np.random.choice(H_vis)
                    x = np.random.choice(W_vis)
                    vis_valid = valid[y, x, 0]
                    vis_visible = occl[y, x, 0] < 0.5
                    if vis_valid and vis_visible:
                        found = True
                        break
                if not found:
                    continue
                radius=5
                cv2.circle(vis1, (x, y), radius=radius, color=color, thickness=-1)
                flow_corr = flow[y, x, :]
                cv2.circle(vis2, (int(x + flow_corr[0]), int(y + flow_corr[1])), radius=radius, color=color, thickness=-1)
            # vis_valid = 255 * einops.repeat(valid[:, :, 0], 'H W -> H W C', C=3)
            # vis_occl = 255 * einops.repeat(occl[:, :, 0], 'H W -> H W C', C=3)
            # vis1[:, :, 0] = vis_occl[:, :, 0]

            vis = np.hstack((
                # vis_valid,
                # vis_occl,
                vis1, vis2))
            stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S_%f")

            cv2.imwrite(str(vis_path / f"{stamp}.png"), vis)
        # assert False, "todo make our shapes compatible with the augmentor..."
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        occl = torch.from_numpy(occl).permute(2, 0, 1).float()

        valid = torch.from_numpy(valid).permute(2, 0, 1).float() > 0.99

        return img1, img2, flow, valid.float(), occl

    def __len__(self):
        return 50000  # placeholder. Can this be fixed?



def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding training set """
    train_dataset = None
    load_occlusion = args.occlusion_module is not None
    if args.dashcam_augmenentation:
        aug_params = {'do_jpeg_transform': True,
                      'do_blend_transform': False,
                      'do_add_text_transform': False,
                      'jpeg_prop': 0.5,
                      }
    else:
        aug_params = {}

    if args.stage == 'chairs':
        aug_params.update({'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True})
        train_dataset = FlyingChairs(aug_params, split='training')

    elif args.stage == 'things':
        aug_params.update({'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True})
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass', load_occlusion=load_occlusion)
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass', load_occlusion=load_occlusion)
        train_dataset = clean_dataset + final_dataset

    elif args.stage == 'sintel_things':
        aug_params.update({'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True})
        things_clean = FlyingThings3D(aug_params, dstype='frames_cleanpass', load_occlusion=load_occlusion)
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean', load_occlusion=load_occlusion)
        things_final = FlyingThings3D(aug_params, dstype='frames_finalpass', load_occlusion=load_occlusion)
        sintel_final = MpiSintel(aug_params, split='training', dstype='final', load_occlusion=load_occlusion)
        train_dataset = 100 * sintel_clean + 100 * sintel_final + things_clean + things_final

    elif args.stage == 'sintel_things_train_subsplit':
        aug_params.update({'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True})
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean', load_occlusion=load_occlusion, subsplit='training')  # noqa 501
        things_final = FlyingThings3D(aug_params, dstype='frames_finalpass', load_occlusion=load_occlusion)
        sintel_final = MpiSintel(aug_params, split='training', dstype='final', load_occlusion=load_occlusion, subsplit='training')  # noqa 501
        train_dataset = 200 * sintel_clean + 200 * sintel_final + things_final

    elif args.stage == 'sintel_things_kubric_train_subsplit':
        aug_params.update({'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True})
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean', load_occlusion=load_occlusion, subsplit='training')  # noqa 501
        things_final = FlyingThings3D(aug_params, dstype='frames_finalpass', load_occlusion=load_occlusion)
        sintel_final = MpiSintel(aug_params, split='training', dstype='final', load_occlusion=load_occlusion, subsplit='training')  # noqa 501

        kubric_aug_params = copy.deepcopy(aug_params)
        kubric_aug_params.update({'min_scale': 1.8, 'max_scale': 2.2, 'stretch_prob': 1.1, 'spatial_aug_prob': 1.1,
                                  'asymmetric_color_aug_prob': 0.0})
        kubric_train = KubricDataset(kubric_aug_params, split='train', load_occlusion=load_occlusion, correct_flow=True)
        train_dataset = 100 * sintel_clean + 100 * sintel_final + things_final + kubric_train

    elif args.stage == 'kubric_long_train_subsplit':
        aug_params.update({'crop_size': args.image_size, 'min_scale': -0.38, 'max_scale': 0.0, 'do_flip': True, 'asymmetric_color_aug_prob': 0.25})
        extended_dataset = 'flow_est'
        kubric_aug_params = copy.deepcopy(aug_params)
        kubric_long_train = KubricLong2024Dataset(kubric_aug_params, split='train', subsplit='train', load_occlusion=load_occlusion, correct_flow=True, extended_dataset=extended_dataset)
        train_dataset = 1000 * kubric_long_train

    elif args.stage == 'kubric_long_train_without_valid_masks_subsplit':
        aug_params.update({'crop_size': args.image_size, 'min_scale': -0.38, 'max_scale': 0.0, 'do_flip': True, 'asymmetric_color_aug_prob': 0.25})
        kubric_aug_params = copy.deepcopy(aug_params)
        load_occl_valid_masks = False
        kubric_long_train = KubricLong2024Dataset(kubric_aug_params, split='train', subsplit='train', load_occlusion=load_occlusion, correct_flow=True, extended_dataset='flow_est', load_occl_valid_masks=load_occl_valid_masks)
        train_dataset = 1000 * kubric_long_train

    elif args.stage == 'kubric_long_debug_subsplit':
        aug_params.update({'crop_size': args.image_size, 'min_scale': -0.38, 'max_scale': 0.0, 'do_flip': True, 'asymmetric_color_aug_prob': 0.25})
        load_occl_valid_masks = args.load_occl_valid_masks
        kubric_aug_params = copy.deepcopy(aug_params)
        kubric_long_train = KubricLong2024Dataset(kubric_aug_params, split='train', subsplit='debug',
                                                  load_occlusion=load_occlusion, correct_flow=True,
                                                  extended_dataset='full',
                                                  load_occl_valid_masks=load_occl_valid_masks)
        train_dataset = 1000000 * kubric_long_train

    elif args.stage == 'sintel':
        aug_params.update({'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True})
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass', load_occlusion=load_occlusion)
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean', load_occlusion=load_occlusion)
        sintel_final = MpiSintel(aug_params, split='training', dstype='final', load_occlusion=load_occlusion)

        if TRAIN_DS == 'C+T+K+S+H':
            kitti_aug_params = deepcopy(aug_params)
            kitti_aug_params.update({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})  # noqa 501
            hd1k_aug_params = deepcopy(aug_params)
            hd1k_aug_params.update({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
            kitti = KITTI(kitti_aug_params)
            hd1k = HD1K(hd1k_aug_params)
            train_dataset = 100 * sintel_clean + 100 * sintel_final + 200 * kitti + 5 * hd1k + things

        elif TRAIN_DS == 'C+T+K/S':
            train_dataset = 100 * sintel_clean + 100 * sintel_final + things

        elif TRAIN_DS == 'C+T+K+S+H+V':
            kitti_aug_params = deepcopy(aug_params)
            kitti_aug_params.update({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})  # noqa 501
            hd1k_aug_params = deepcopy(aug_params)
            hd1k_aug_params.update({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
            kitti = KITTI(kitti_aug_params)
            hd1k = HD1K(hd1k_aug_params)
            train_dataset = 100 * sintel_clean + 100 * sintel_final + 200 * kitti + 5 * hd1k + things

    elif args.stage == 'kitti':
        aug_params.update({'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False})
        train_dataset = KITTI(aug_params, split='training')

    elif args.stage == 'viper':
        viper_aug_params = deepcopy(aug_params)
        viper_aug_params.update({'crop_size': args.image_size, 'min_scale': -2, 'max_scale': -0.5, 'do_flip': True})
        train_dataset = VIPER(viper_aug_params, split='training')

    elif args.stage == 'all':
        aug_params.update({'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True})
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass', load_occlusion=load_occlusion)
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean', load_occlusion=load_occlusion)
        sintel_final = MpiSintel(aug_params, split='training', dstype='final', load_occlusion=load_occlusion)
        kitti_aug_params = deepcopy(aug_params)
        kitti_aug_params.update({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
        hd1k_aug_params = deepcopy(aug_params)
        hd1k_aug_params.update({'crop_size': args.image_size, 'min_scale': -2, 'max_scale': -0.5, 'do_flip': True})
        viper_aug_params = deepcopy(aug_params)
        viper_aug_params.update({'crop_size': args.image_size, 'min_scale': -2, 'max_scale': -0.5, 'do_flip': True})
        kitti = KITTI(kitti_aug_params, split='training')
        hd1k = HD1K(hd1k_aug_params)
        viper = VIPER(viper_aug_params, split='training')
        # sintel 0.4
        # KITTI 0.2
        # VIPER 0.2 down 2x
        # HD1K 0.08 down 2x
        # Things 0.12
        # kitti, viper, hd1k - reduced probability of spatial aut to 0.5
        train_dataset = (200 * sintel_clean) + (200 * sintel_final) + (200 * kitti) + (80 * hd1k) + (200 * viper) + (120 * things)  # noqa 501

    num_workers = getattr(args, 'n_workers', 16)
    console.print(f'[blue]Number of dataloader workers: {num_workers}')

    shuffle = not getattr(args, 'no_shuffle', False)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, #timeout=300,
                                   pin_memory=False, shuffle=shuffle, num_workers=num_workers, drop_last=True)

    console.print(f'[blue]Training with {len(train_dataset)} image pairs')
    return train_loader


def combine_datasets_with_weights(weight_dataset_pairs):
    multipliers = np.array([weight / len(dataset) for weight, dataset in weight_dataset_pairs])
    multipliers /= np.amin(multipliers)
    multipliers = np.round(multipliers).astype(np.int32).tolist()
    print(f"Datasets combined with multipliers: {multipliers}")
    datasets = [dataset for weight, dataset in weight_dataset_pairs]
    lengths = [len(dataset) for dataset in datasets]
    print(f"result in sample counts (true): {[mult * length for mult, length in zip(multipliers, lengths)]}")
    print(f"result in sample counts (virtual): {[mult * 100 * length for mult, length in zip(multipliers, lengths)]}")
    weighted_datasets = [int(mult) * 100 * dataset for mult, dataset in zip(multipliers, datasets)]

    result = weighted_datasets[0]
    for dataset in weighted_datasets[1:]:
        result += dataset

    return result


if __name__ == '__main__':
    if False:  # debug visualization {{{
        from timeit import default_timer as timer
        aug_params = None
        # aug_params = {}
        # aug_params.update({'crop_size': (700, 700), 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True})
        # aug_params.update({'crop_size': (1000, 1000), 'min_scale': 1, 'max_scale': 1.0001, 'do_flip': True})
        ds = KubricLongflowDataset(aug_params)
        ds.max_delta = 3
        loader = data.DataLoader(ds, batch_size=3,
                                 pin_memory=False, shuffle=True, num_workers=0, drop_last=True)
        import sys
        from PIL import Image
        from torchvision.transforms import ColorJitter
        photo_aug = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        start = timer()
        n_samples = 0
        for i, sample in enumerate(loader):
            if i >= 20:
                break
            n_samples += 1

            image1, image2, flow, valid, occl = [x for x in sample]
            if True:
                cv2.namedWindow("cv: img1", cv2.WINDOW_NORMAL)
                img1 = einops.rearrange(image1.cpu().numpy(), 'B C H W -> H (B W) C', C=3)[:, :, ::-1]
                img1 = np.round(img1).astype(np.uint8)

                cv2.namedWindow("cv: img2", cv2.WINDOW_NORMAL)
                img2 = einops.rearrange(image2.cpu().numpy(), 'B C H W -> H (B W) C', C=3)[:, :, ::-1]
                img2 = np.round(img2).astype(np.uint8)

                cv2.namedWindow("cv: aug_img1", cv2.WINDOW_NORMAL)
                for aug_it in range(100):
                    if aug_it == 0:
                        aug_img1 = img1.copy()
                    else:
                        aug_img1 = np.array(photo_aug(Image.fromarray(img1[:, :, ::-1])), dtype=np.uint8)[:, :, ::-1]
                    cv2.imshow("cv: aug_img1", aug_img1)
                    while True:
                        c = cv2.waitKey(0)
                        if c == ord(' '):
                            break
                        elif c == ord('q'):
                            sys.exit(1)

                cv2.imshow("cv: img1", img1)
                cv2.imshow("cv: img2", img2)
                while True:
                    c = cv2.waitKey(0)
                    if c == ord('q'):
                        sys.exit(1)
                    elif c == ord(' '):
                        break

            print(f'sample length: {len(sample)}')
        print(f'{(timer() - start) / n_samples} s/sample')
        # }}}
