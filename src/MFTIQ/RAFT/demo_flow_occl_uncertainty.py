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

from RAFT.core.datasets import MpiSintel
from RAFT.core.utils import flow_viz
from RAFT.core.utils.utils import InputPadder
from RAFT.core.utils.flow_viz import flow_to_color

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

class ImageSaver():
    def __init__(self, root_path, *args, **kwargs):
        self.root_path = root_path
        self.index = 0
        self.max_sigma = 0

    def torch2numpy(self, data):
        return data.detach().cpu().numpy().transpose(1,2,0)

    def save_image(self, data, to_int=False, to255=False):
        pass

    def normalise(self, data, values01=False, to_uint8=False, to255=False, to3ch=False):
        data_shape = data.shape
        if values01:
            dmin = data.min()
            dmax = data.max()
            data = (data - dmin) / (dmax - dmin)
        data = data * 255. if to255 else data
        data = data.astype(np.uint8) if to_uint8 else data
        if to3ch:
            if len(data_shape) == 2:
                data = np.stack([data] * 3, axis=2)
            elif data_shape[2] == 1:
                data = np.concatenate([data] * 3, axis=2)
        return data

    def addtext2img(self, img, text, color=(255, 255, 255)):
        img = cv2.putText(img, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, color=color, thickness=3)
        return img

    def __call__(self, im1, im2, flow_gt, flow_pred, occl_gt, occl_pred, uncertainty_pred, im1_path=None, **kwargs):
        if im1_path[0] == '/':
            im1_path = im1_path[1:]

        im1_np = self.torch2numpy(im1)
        im2_np = self.torch2numpy(im2)
        occl_gt_np =  self.torch2numpy(occl_gt)
        occl_valid_np = np.logical_and(occl_gt_np > 1, occl_gt_np < 254) * 1
        occl_pred_np = self.torch2numpy(occl_pred)
        occl_pred_thresh_np = (occl_pred_np > 0.5) * 1

        uncertainty_pred_np = self.torch2numpy(uncertainty_pred)
        flow_gt_np = self.torch2numpy(flow_gt)
        flow_pred_np = self.torch2numpy(flow_pred)

        flow_gt_img = flow_to_color(flow_gt_np)
        flow_pred_img = flow_to_color(flow_pred_np)

        sigma_np = np.sqrt(np.exp(uncertainty_pred_np))
        if self.max_sigma < np.max(sigma_np):
            self.max_sigma = np.max(sigma_np)
        stat_sigma = f'Sigma - OverallMax: {self.max_sigma} Max: {np.max(sigma_np)}, Min: {np.min(sigma_np)}, Q95: {np.quantile(sigma_np, 0.95)}, Q99: {np.quantile(sigma_np, 0.99)}, Q995: {np.quantile(sigma_np, 0.995)}, Q999: {np.quantile(sigma_np, 0.999)}'
        print(stat_sigma)

        stat_filepath = os.path.join(self.root_path, 'statistic.txt')
        with open(stat_filepath, 'a') as f:
            f.write(stat_sigma+'\n')

        flow_sad = np.sum(np.abs(flow_gt_np - flow_pred_np), axis=2, keepdims=True)
        flow_ssd = np.sum((flow_gt_np - flow_pred_np)**2, axis=2, keepdims=True)
        flow_epe = np.sqrt(flow_ssd)

        flow_epe_norm = self.normalise(flow_epe, to255=False, values01=True)
        flow_sad_norm = self.normalise(flow_sad, to255=False, values01=True)
        flow_ssd_norm = self.normalise(flow_ssd, to255=False, values01=True)
        sigma_norm_np = self.normalise(sigma_np, to255=False, values01=True)
        ssim_sigma_epe = ssim(flow_epe_norm[:,:,0], sigma_norm_np[:,:,0], full=True, win_size=3, gaussian_weights=True)
        ssim_sigma_sad = ssim(flow_sad_norm[:, :, 0], sigma_norm_np[:, :, 0], full=True, win_size=3, gaussian_weights=True)
        ssim_sigma_ssd = ssim(flow_ssd_norm[:, :, 0], sigma_norm_np[:, :, 0], full=True, win_size=3, gaussian_weights=True)
        ssim_sigma_epe_img = self.normalise(np.abs(ssim_sigma_epe[1]), to_uint8=True, to255=True, values01=False, to3ch=True)
        ssim_sigma_epe_img = self.addtext2img(ssim_sigma_epe_img, f'SSIM sigma/epe {ssim_sigma_epe[0]:04f}', color=(255,0,0))
        ssim_sigma_sad_img = self.normalise(np.abs(ssim_sigma_sad[1]), to_uint8=True, to255=True, values01=False, to3ch=True)
        ssim_sigma_sad_img = self.addtext2img(ssim_sigma_sad_img, f'SSIM sigma/sad {ssim_sigma_sad[0]:04f}', color=(255, 0, 0))
        ssim_sigma_ssd_img = self.normalise(np.abs(ssim_sigma_ssd[1]), to_uint8=True, to255=True, values01=False, to3ch=True)
        ssim_sigma_ssd_img = self.addtext2img(ssim_sigma_ssd_img, f'SSIM sigma/ssd {ssim_sigma_ssd[0]:04f}', color=(255, 0, 0))

        im1_img = self.addtext2img(self.normalise(im1_np, to_uint8=True), 'im1')
        im2_img = self.addtext2img(self.normalise(im2_np, to_uint8=True), 'im2')
        flow_gt_img = self.addtext2img(flow_gt_img, 'OF-GT')
        flow_pred_img = self.addtext2img(flow_pred_img, 'OF-EST')

        occl_diff = np.abs(occl_gt_np.astype(float) - occl_pred_thresh_np*255)
        occl_gt_np[occl_valid_np!=0] = 120
        occl_gt_img = self.addtext2img(self.normalise(occl_gt_np, to3ch=True, to_uint8=True), 'OCCL GT', color=(255,0,0))
        # occl_diff[occl_valid_np!=0] = 120
        occl_diff_img = self.addtext2img(self.normalise(occl_diff, to3ch=True, to_uint8=True), 'OCCL DIFF', color=(255,0,0))


        occl_pred_img = self.normalise(occl_pred_thresh_np, to3ch=True, to255=True, to_uint8=True)
        occl_pred_img = self.addtext2img(occl_pred_img, 'OCCL EST', color=(255,0,0))
        flow_sad_img = self.normalise(flow_sad, to_uint8=True, to255=True, values01=True, to3ch=True)
        flow_sad_img = self.addtext2img(flow_sad_img, 'OF-SAD', color=(255,0,0))
        flow_ssd_img = self.normalise(flow_ssd, to_uint8=True, to255=True, values01=True, to3ch=True)
        flow_ssd_img = self.addtext2img(flow_ssd_img, 'OF-SSD', color=(255,0,0))
        flow_epe_img = self.normalise(flow_epe, to_uint8=True, to255=True, values01=True, to3ch=True)
        flow_epe_img = self.addtext2img(flow_epe_img, 'OF-EPE', color=(255,0,0))
        sigma_np_img = self.normalise(sigma_np, to_uint8=True, to255=True, values01=True, to3ch=True)
        sigma_np_img = self.addtext2img(sigma_np_img, 'SIGMA', color=(255,0,0))

        # comp for flow
        flow_comp_row1 = np.concatenate([im1_img, im2_img, flow_gt_img], axis=1)
        flow_comp_row2 = np.concatenate([flow_sad_img, flow_epe_img, flow_pred_img], axis=1)
        flow_comp_img = np.concatenate([flow_comp_row1, flow_comp_row2], axis=0)
        flow_img_name = os.path.join(self.root_path, im1_path.replace('.png', '_flow.jpg'))

        # comp for sigma
        sigma_comp_row1 = np.concatenate([flow_gt_img, flow_pred_img, sigma_np_img], axis=1)
        sigma_comp_row2 = np.concatenate([flow_sad_img, flow_epe_img, flow_ssd_img], axis=1)
        sigma_comp_row3 = np.concatenate([ssim_sigma_sad_img, ssim_sigma_epe_img, ssim_sigma_ssd_img], axis=1)
        sigma_comp_img = np.concatenate([sigma_comp_row1, sigma_comp_row2, sigma_comp_row3], axis=0)
        sigma_img_name = os.path.join(self.root_path, im1_path.replace('.png', '_sigma.jpg'))

        # comp for occlusion
        occl_comp_img = np.concatenate([occl_gt_img, occl_pred_img, occl_diff_img], axis=1)
        occl_img_name = os.path.join(self.root_path, im1_path.replace('.png', '_occl.jpg'))

        mkdir_from_full_file_path_if_not_exist(flow_img_name)
        cv2.imwrite(flow_img_name, cv2.cvtColor(flow_comp_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(sigma_img_name, cv2.cvtColor(sigma_comp_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(occl_img_name, cv2.cvtColor(occl_comp_img, cv2.COLOR_RGB2BGR))


def main(args):
    # Load RAFT
    device = f'cuda:{args.gpus}'

    raft_kwargs = {
        'occlusion_module': 'separate_with_uncertainty',
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

    # saver for image
    imsaver = ImageSaver(args.save_path)

    # DATASET WITH GT FLOW and OCCL
    sintel_clean = MpiSintel(split='training', dstype='clean', load_occlusion=True)
    sintel_final = MpiSintel(split='training', dstype='final', load_occlusion=True)

    datasets = {'sintel_clean': sintel_clean, 'sintel_final': sintel_final}
    for dataset_name, data in datasets.items():
        print(f'Processing dataset {dataset_name}:')
        for idx in tqdm.trange(len(data)):
            im1o, im2o, flow_gt, valid, occl_gt = data[idx]
            c_im1_path = data.get_reference_frame_path(idx, relative=True)

            im1 = torch.unsqueeze(im1o, 0)
            im2 = torch.unsqueeze(im2o, 0)

            padder = InputPadder(im1.shape, mode='sintel')
            im1, im2 = padder.pad(im1, im2)

            all_predictions = raft(im1.to(device), im2.to(device), iters=args.iters, test_mode=True)
            flow_pred = torch.squeeze(padder.unpad(all_predictions['flow']), dim=0)
            occl_pred = torch.squeeze(padder.unpad(all_predictions['occlusion'].softmax(dim=1)[:,1:2,:,:]), dim=0)
            uncertainty_pred = torch.squeeze(padder.unpad(all_predictions['uncertainty']), dim=0)

            imsaver(im1o, im2o, flow_gt, flow_pred, occl_gt, occl_pred, uncertainty_pred, im1_path=c_im1_path)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/home.stud/neoramic/repos/raft_new_debug/RAFT/checkpoints/50000_raft-things-sintel-occlusion-uncertainty.pth', help="restore checkpoint")
    parser.add_argument('--save_path', default='/datagrid/personal/neoral/RAFT_occl_uncertainty_outputs', help="saving_path")
    parser.add_argument('--iters', type=int, default=24, help='number of iterations')
    parser.add_argument('--gpus', type=int, default=0, help='gpu number')
    args = parser.parse_args()

    main(args)