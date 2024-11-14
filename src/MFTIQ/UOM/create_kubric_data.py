# MFTIQ - WACV2025
import torch

from MFTIQ.RAFT.multiflow_from_kubric import get_multiflow
from MFTIQ.config import Config, load_config
from MFTIQ.utils.io import write_flowou, write_normals, read_normals, read_flowou
from MFTIQ.UOM.utils.image_manipulation import flow_to_coords, sample_with_coords

import time
import numpy as np
import einops
import os
import matplotlib
import argparse
import torch
from pathlib import Path
import uuid
from PIL import Image
import pickle
import random
from enum import Enum

import cv2 as cv

if os.getenv('REMOTE_DEBUG'):
    matplotlib.use('module://backend_interagg')
print(f'{matplotlib.get_backend() = }')

import matplotlib.pyplot as plt
from MFTIQ.RAFT.core.utils.flow_viz import flow_to_color, flow_to_image


class CompStatus(Enum):
    FAILED = 'failed'
    FLOW_EST_OK = 'flow_estimated_successfully'
    PROCESSING = 'processing'
    NOT_COMPUTED = 'not_computed'
    FINISHED = 'finished'
    CHECKED = 'checked'

def save_data_creation_config(sequence_path, idx_list, save_path, status=None):
    save_path.mkdir(parents=True, exist_ok=True)
    config_path = save_path / 'data_creation_cfg.pkl'
    status = status if status is not None else CompStatus.FAILED
    data_creation_config = {'status': status, 'sequence_path': sequence_path, 'idx_list': idx_list,
                            'save_path': save_path}
    with open(config_path, 'wb') as f:
        pickle.dump(data_creation_config, f)


def load_data_creation_config(save_path):
    config_path = save_path / 'data_creation_cfg.pkl'
    try:
        with open(config_path, 'rb') as f:
            data_creation_config = pickle.load(f)
        status = data_creation_config['status']
        sequence_path = data_creation_config['sequence_path']
        idx_list = data_creation_config['idx_list']
        save_path = data_creation_config['save_path']
    except FileNotFoundError:
        status = CompStatus.FAILED
        sequence_path = None
        idx_list = None
        save_path = None
    return sequence_path, idx_list, save_path, status

def get_flower():
    conf = Config()
    conf.flow_config = load_config('configs/flow/FlowFormerPP_sintel.py')
    flower = conf.flow_config.of_class(conf.flow_config)
    return flower

def main(args):
    rng_main = np.random.RandomState(seed=args.seed)
    max_rand_int = 2**31

    flower = get_flower()

    for i in range(args.samples_start):
        _ = np.random.RandomState(rng_main.randint(0, max_rand_int))

    for i in range(args.samples_start, args.samples_end+1):
        rng_c = np.random.RandomState(rng_main.randint(0, max_rand_int))
        sequence_idx = rng_c.randint(args.subdir_start, args.subdir_end + 1)
        sequence_idx_str = f'{sequence_idx:05d}'

        sequence_length = rng_c.randint(args.sequence_length_min, args.sequence_length_max + 1)
        start_frame_idx = rng_c.randint(args.trim_start, args.trim_end - sequence_length)
        end_frame_idx = start_frame_idx + sequence_length

        idx_list = range(start_frame_idx, end_frame_idx, args.step_size)
        if rng_c.rand() > 0.5:
            idx_list = reversed(idx_list)
        idx_list = list(idx_list)

        sequence_path = args.dataroot / sequence_idx_str / args.subdir

        save_id_str = f'{sequence_idx_str}_{idx_list[0]:04d}_{idx_list[-1]:04d}'
        save_path = args.saveroot / save_id_str

        save_data_creation_config(sequence_path, idx_list, save_path, CompStatus.NOT_COMPUTED)

        data = est_flow_data(sequence_path, flower, idx_list)
        if data is None:
            save_data_creation_config(sequence_path, idx_list, save_path, status=CompStatus.FAILED)
        else:
            save_flow_data(data, save_path)
            save_data_creation_config(sequence_path, idx_list, save_path, status=CompStatus.FLOW_EST_OK)


        print(sequence_path)
        print(save_id_str)


def est_flow_data(data_path, flower, idx_list):
    data2save = {}

    idx_list = [idx_list[0], idx_list[-2], idx_list[-1]]
    try:
        kubric_data = get_multiflow(data_path, frames=idx_list)
    except:
        return None

    flow_last, _ = flower.compute_flow(kubric_data['rgb'][0],
                                       kubric_data['rgb'][-1],
                                       mode='flow',
                                       init_flow=kubric_data['flow'][-1])
    flow_before_last, _ = flower.compute_flow(kubric_data['rgb'][0],
                                              kubric_data['rgb'][-2],
                                              mode='flow',
                                              init_flow=kubric_data['flow'][-2])

    flow_last_np = flow_last.detach().cpu().numpy()
    flow_before_last_np = flow_before_last.detach().cpu().numpy()
    data2save['flow_last'] = flow_last_np
    data2save['flow_before_last'] = flow_before_last_np
    return data2save


def second_stage(args):
    root = args.saveroot
    all_dirs = list(root.iterdir())

    random.seed(args.seed)
    random.shuffle(all_dirs)

    for c_dir in all_dirs:
        sequence_path, idx_list, save_path, status = load_data_creation_config(c_dir)
        if status != CompStatus.FLOW_EST_OK:
            continue
        save_data_creation_config(sequence_path, idx_list, save_path, status=CompStatus.PROCESSING)

        data = create_data(sequence_path, save_path, idx_list)
        if data is None:
            save_data_creation_config(sequence_path, idx_list, save_path, status=CompStatus.FAILED)
            continue

        save_data(data, save_path)
        save_data_creation_config(sequence_path, idx_list, save_path, status=CompStatus.FINISHED)

        print(sequence_path)


def check_stage(args):
    root = args.saveroot
    trash_root = args.saveroot.parent / f'{args.saveroot.name}_trash'
    all_dirs = list(root.iterdir())

    random.seed(args.seed)
    random.shuffle(all_dirs)

    for c_dir in all_dirs:
        sequence_path, idx_list, save_path, status = load_data_creation_config(c_dir)
        if status != CompStatus.FINISHED:
            continue
        save_data_creation_config(sequence_path, idx_list, save_path, status=CompStatus.PROCESSING)
        try:
            for c_file in save_path.iterdir():
                if 'flowouX16.pkl' in str(c_file):
                    read_flowou(c_file)
                elif 'normalX16.pkl' in str(c_file):
                    read_normals(c_file)
        except:
            save_data_creation_config(sequence_path, idx_list, save_path, status=CompStatus.FAILED)

            trash_root.mkdir(parents=True, exist_ok=True)
            save_path.rename(trash_root / save_path.name)
            continue

        save_data_creation_config(sequence_path, idx_list, save_path, status=CompStatus.CHECKED)

        print(sequence_path)


def save_flow_data(data, save_path):
    save_path.mkdir(parents=True, exist_ok=True)
    suffix_flow = '.flowouX16.pkl'
    write_flowou(save_path / f'flow_last_tmp{suffix_flow}', flow=data['flow_last'], occlusions=None, uncertainty=None)
    write_flowou(save_path / f'flow_before_last_tmp{suffix_flow}', flow=data['flow_before_last'], occlusions=None, uncertainty=None)


def save_data(data, save_path):
    save_path.mkdir(parents=True, exist_ok=True)

    suffix_flow = '.flowouX16.pkl'
    suffix_normal = '.normalX16.pkl'

    write_flowou(save_path / f'flow_gt{suffix_flow}', flow=data['flow_gt'], occlusions=data['occl_gt'], uncertainty=data['occl_valid'])
    write_flowou(save_path / f'flow_est{suffix_flow}', flow=data['flow_est'], occlusions=None, uncertainty=None)
    write_normals(save_path / f'normal_reference{suffix_normal}', normals=data['normal_reference'], mask=data['foreground_mask'])
    write_normals(save_path / f'normal_target{suffix_normal}', normals=data['normal_target'], mask=data['foreground_mask'])

    im = Image.fromarray(data['rgb_reference'])
    im.save(save_path / 'rgb_reference.png')

    im = Image.fromarray(data['rgb_target'])
    im.save(save_path / 'rgb_target.png')

    return 0

def compute_occl_valid(data, cidx=-1):

    coords1, coords2 = flow_to_coords(einops.rearrange(torch.from_numpy(data['flow'][cidx]), 'C H W -> 1 H W C'))
    warped_segmentation = sample_with_coords(einops.rearrange(torch.from_numpy(data['segmentations'][cidx] + 1).to(torch.float32), 'C H W -> 1 C H W'), coords2)
    warped_segmentation = warped_segmentation[0].detach().cpu().numpy() - 1
    seg_diff_threshold = np.abs(warped_segmentation[0] - data['segmentations'][0][0].astype(float)) > 0.01

    c_occl = data['occlusion'][cidx][0]

    c_occl_from_depth1 = data['depth_proj'][cidx][0] * 0.99 > data['depth_est'][cidx][0]
    c_occl_from_depth2 = data['depth_proj'][cidx][0] * 1.01 < data['depth_est'][cidx][0]

    c_coords2 = einops.rearrange(coords2.detach().cpu().numpy(), '1 H W C -> H W C')
    if c_coords2.shape[0] != c_coords2.shape[1]:
        raise NotImplementedError('Not implemented for non-square images')
    c_flow_pointing_outside = c_coords2 < 0.0
    c_flow_pointing_outside = np.logical_or(c_flow_pointing_outside, c_coords2 > (c_coords2.shape[0]-1))
    c_flow_pointing_outside = np.logical_or(c_flow_pointing_outside[:,:,0], c_flow_pointing_outside[:,:,1])

    # non-valid = occl_gt says NONOCCL, but mask from depth says OCCL
    non_valid1 = np.logical_and(np.logical_not(c_occl), c_occl_from_depth1)
    non_valid2 = np.logical_and(np.logical_not(c_occl), c_occl_from_depth2)
    non_valid3 = np.logical_and(np.logical_not(c_occl), seg_diff_threshold)
    non_valid4 = np.logical_and(np.logical_not(c_occl), c_flow_pointing_outside)

    valid_correction = np.logical_and(c_occl, c_flow_pointing_outside)
    non_valid = np.logical_or(np.logical_or(non_valid1, non_valid2), np.logical_or(non_valid3, non_valid4))
    non_valid[valid_correction] = False

    kernel = np.ones((3, 3), np.uint8)
    dilation_non_valid = cv.dilate(non_valid.astype(np.uint8), kernel, iterations=1)
    dilation_non_valid = einops.rearrange(dilation_non_valid, 'H W -> 1 H W')
    valid = np.logical_not(dilation_non_valid)

    return valid


def check_valid_correspondences(occl, occl_valid, foreground_mask,
                                foreground_occl_area_threshold=0.25,
                                background_occl_area_threshold=0.25):
    foreground_area = np.sum(foreground_mask)
    background_area = np.sum(np.logical_not(foreground_mask))

    foreground_nonoccl_area = np.sum(np.logical_and(np.logical_and(np.logical_not(occl), occl_valid), foreground_mask))
    background_nonoccl_area = np.sum(np.logical_and(np.logical_and(np.logical_not(occl), occl_valid), np.logical_not(foreground_mask)))

    if foreground_nonoccl_area / foreground_area < foreground_occl_area_threshold:
        return False
    if background_nonoccl_area / background_area < background_occl_area_threshold:
        return False
    return True


def create_data(data_path, save_path, idx_list):
    data2save = {}
    try:
        kubric_data = get_multiflow(data_path, frames=idx_list)
    except:
        return None
    data2save['rgb_reference'] = kubric_data['rgb'][0]
    data2save['rgb_target'] = kubric_data['rgb'][-1]

    occl_valid = compute_occl_valid(kubric_data)
    data2save['occl_valid'] = occl_valid

    foreground_mask = kubric_data['segmentations'][0] > 0
    data2save['foreground_mask'] = foreground_mask
    data2save['normal_reference'] = kubric_data['normals'][0]
    data2save['normal_target'] = kubric_data['normals'][-1]

    occl = kubric_data['occlusion'][-1]
    if not check_valid_correspondences(occl, occl_valid, foreground_mask):
        return None

    suffix_flow = '.flowouX16.pkl'
    flow_last_np, _, _ = read_flowou(save_path / f'flow_last_tmp{suffix_flow}')
    flow_before_last_np, _, _ = read_flowou(save_path / f'flow_before_last_tmp{suffix_flow}')

    flow_subset_for_avg_speed = kubric_data['flow'][:-1]
    flow_subset_for_avg_speed.append(flow_before_last_np)

    occl_subset_for_not_matched = kubric_data['occlusion'][:-1]

    data2save['flow_est'] = flow_last_np
    data2save['flow_gt'] = kubric_data['flow'][-1]
    data2save['occl_gt'] = kubric_data['occlusion'][-1]
    return data2save


def parse_arguments():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataroot', type=Path, default=Path('/datagrid/tlabdata/neoramic/mft2024/kubric_datasets/test_panning/RES_1024x1024'))
    parser.add_argument('--subdir', type=str, default=Path('FPS_120__NFRAMES_240__CAM_linear_movement_linear_lookat__GE_forward_backward_cycle__camshake__TYPE_012'))
    parser.add_argument('--saveroot', type=Path, default=Path('/datagrid/tlabdata/neoramic/mft2024/kubric_datasets/20240410_002_030/'))
    parser.add_argument('--gpuid', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--step_size', type=int, default=1)

    parser.add_argument('--samples_start', type=int, default=1, help='How many training samples may be generated per single run')
    parser.add_argument('--samples_end', type=int, default=100, help='How many training samples may be generated per single run')

    parser.add_argument('--trim_start', type=int, default=3, help='First X frames will be ignored')
    parser.add_argument('--trim_end', type=int, default=235, help='Up to frame number X will be used, rest ignored')

    parser.add_argument('--subdir_start', type=int, default=1)
    parser.add_argument('--subdir_end', type=int, default=168)

    parser.add_argument('--sequence_length_min', type=int, default=2, help='minimum length of the sequence')
    parser.add_argument('--sequence_length_max', type=int, default=30, help='maximum length of the sequence')

    parser.add_argument('--second_stage', action='store_true', help='run final stage with flowformer and save final outputs')
    parser.add_argument('--check_stage', action='store_true', help='check data after second stage. Move to thrash directory invalid files')

    parser.add_argument('--debug', action='store_true', help='debugging data')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    if args.gpuid is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpuid}'
    if args.second_stage:
        assert not args.check_stage
        second_stage(args)
        check_stage(args)
    else:
        main(args)
