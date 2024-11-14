from __future__ import print_function, division
import sys, os
from RAFT.core.raft import RAFT
import tqdm

# sys.path.append('core')

import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from RAFT.core.datasets import MpiSintel, KubricDataset
from RAFT.core.utils import flow_viz
from RAFT.core.utils.utils import InputPadder
from RAFT.core.utils.flow_viz import flow_to_color
import einops

from RAFT.core.utils.statistics import FlowOUStatistics, OcclDistStatistics, OcclEPESigmaStatistics, MultiflowEPESigmaStatistics

import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def mkdir_if_not_exist(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def mkdir_from_full_file_path_if_not_exist(path):
    basename = os.path.basename(path)
    mkdir_if_not_exist(path[:-len(basename)])

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__.update(kwargs)
def main(args):
    # Load RAFT
    device = f'cuda:{args.gpus}'

    raft_kwargs = {
        'occlusion_module': 'separate_with_uncertainty' if args.occlusion_module is None else args.occlusion_module,
        'restore_ckpt': args.model,
        'small': False,
        'mixed_precision': False,
    }
    raft_params = AttrDict(**raft_kwargs)

    # Of course, you can use argparse instead of attr_dict
    raft = torch.nn.DataParallel(RAFT(raft_params))
    raft.load_state_dict(torch.load(args.model, map_location='cpu'))

    raft = raft.module
    raft.requires_grad_(False)
    raft.to(device)
    raft.eval()

    # DATASET WITH GT FLOW and OCCL
    kubric_validation = KubricDataset(split='validation', load_occlusion=True, correct_flow=True)
    datasets = {'kubric_val_correct_minus_plus_flow': kubric_validation,}

    delta_max = 24

    for dataset_name, data in datasets.items():
        print(f'Processing dataset {dataset_name}:')

        graphs_save_path = os.path.join(args.save_path, os.path.basename(args.model)[:-4])
        mkdir_if_not_exist(graphs_save_path)

        # flowouStats = FlowOUStatistics(dataset_name, save_root=graphs_save_path, model_name=os.path.basename(args.model)[:-4])
        # occlStats = OcclEPESigmaStatistics(dataset_name, save_root=graphs_save_path, nbins=20, model_name=os.path.basename(args.model)[:-4])
        # occlDistStats = OcclDistStatistics(dataset_name, save_root=graphs_save_path, nbins=16, model_name=os.path.basename(args.model)[:-4])
        #
        # flowouStats_list = []
        # occlStats_list = []
        # occlDistStats_list = []
        #
        # for delta in range(1, delta_max):
        #     flowouStats_list.append(FlowOUStatistics(f'delta_{delta:02d}_{dataset_name}', save_root=graphs_save_path, model_name=f'delta: {delta:02d} {os.path.basename(args.model)[:-4]}'))
        #     occlStats_list.append(OcclEPESigmaStatistics(f'delta_{delta:02d}_{dataset_name}', save_root=graphs_save_path, nbins=20, model_name=f'delta: {delta:02d} {os.path.basename(args.model)[:-4]}'))
        #     occlDistStats_list.append(OcclDistStatistics(f'delta_{delta:02d}_{dataset_name}', save_root=graphs_save_path, nbins=16, model_name=f'delta: {delta:02d} {os.path.basename(args.model)[:-4]}'))
        multiepeStats = MultiflowEPESigmaStatistics(dataset_name, save_root=graphs_save_path, model_name=os.path.basename(args.model)[:-4], ndelta=23)

        for idx in tqdm.trange(len(data.multi_image_list)):

            im1o0, im2o0, flow_gt0, valid0, occl_gt0 = data.get_data_delta(idx, 0)
            obj_gt_template = flow_gt0[0:1, :, :] > 0.2

            for delta in range(1,delta_max):
                im1o, im2o, flow_gt, valid, occl_gt = data.get_data_delta(idx, delta)
                c_im1_path = data.get_reference_frame_path(idx, relative=True)

                im1 = torch.unsqueeze(im1o, 0)
                im2 = torch.unsqueeze(im2o, 0)

                padder = InputPadder(im1.shape, mode='sintel')
                im1, im2 = padder.pad(im1, im2)

                all_predictions = raft(im1.to(device), im2.to(device), iters=args.iters, test_mode=True)
                flow_pred = torch.squeeze(padder.unpad(all_predictions['flow']), dim=0)
                occl_pred = torch.squeeze(padder.unpad(all_predictions['occlusion'].softmax(dim=1)[:,1:2,:,:]), dim=0)
                uncertainty_pred = torch.squeeze(padder.unpad(all_predictions['uncertainty']), dim=0)

                # flowouStats.add_data(flow_pred, flow_gt, valid, occl_pred, occl_gt, uncertainty_pred)
                # occlStats.add_data(flow_pred, flow_gt, valid, occl_pred, occl_gt, uncertainty_pred)
                # occlDistStats.add_data(flow_pred, flow_gt, valid, occl_pred, occl_gt, uncertainty_pred)
                #
                # flowouStats_list[delta-1].add_data(flow_pred, flow_gt, valid, occl_pred, occl_gt, uncertainty_pred)
                # occlStats_list[delta-1].add_data(flow_pred, flow_gt, valid, occl_pred, occl_gt, uncertainty_pred)
                # occlDistStats_list[delta-1].add_data(flow_pred, flow_gt, valid, occl_pred, occl_gt, uncertainty_pred)

                multiepeStats.add_data(flow_pred, flow_gt, valid, occl_pred, occl_gt, uncertainty_pred, delta=delta, fg_mask=obj_gt_template, correct_flow=False)

            if args.debug:
                if idx >= 20:
                    break

        multiepeStats.save_graphs()
        # occlStats.save_graphs()
        # occlDistStats.save_graphs()
        # flowouStats.save_graphs(debug=args.debug)
        #
        # for delta in range(1, delta_max):
        #     occlStats_list[delta-1].save_graphs()
        #     occlDistStats_list[delta-1].save_graphs()
        #     flowouStats_list[delta-1].save_graphs(debug=args.debug)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/home.stud/neoramic/repos/raft_new_debug/RAFT/checkpoints/50000_raft-things-sintel-occlusion-uncertainty-base-sintel.pth', help="restore checkpoint")
    parser.add_argument('--save_path', default='/datagrid/personal/neoral/RAFT_occl_uncertainty_outputs', help="saving_path")
    parser.add_argument('--iters', type=int, default=24, help='number of iterations')
    parser.add_argument('--gpus', type=int, default=0, help='gpu number')
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--occlusion_module', type=str, default=None,
                        choices=[None, 'separate', 'with_uncertainty', 'separate_with_uncertainty',
                                 'separate_with_uncertainty_upsample8',
                                 'separate_with_uncertainty_upsample8_morelayers'])
    parser.add_argument('--subsplit', type=str, default=None, choices=[None, 'training', 'validation'])

    args = parser.parse_args()

    main(args)