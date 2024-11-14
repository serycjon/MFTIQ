import sys
sys.path.append("/datagrid/personal/neoral/repos/raft_debug")
sys.path.append('core')

import argparse
import os
import numpy as np
import torch
from PIL import Image

from RAFT.core import RAFT
import RAFT.core.utils.flow_gen as flow_gen
from tqdm import tqdm

from RAFT.core.utils import InputPadder

DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img_torch = torch.from_numpy(img).permute(2, 0, 1).float()
    return img_torch[None].to(DEVICE), img

def gen(args):

    for dataset_name in ['training']: #, 'test']:
        for dataset_subname in ['clean', 'final']:

            data_root = '/datagrid/public_datasets/Sintel-complete/{:s}/{:s}'.format(dataset_name, dataset_subname)
            model_name = args.model.split('/')[-1][:-4]
            save_root = '/datagrid/personal/neoral/datasets/optical_flow_neomoseg/raft_new_export/sintel_{:s}/{:s}/{:s}'.format(dataset_subname,model_name,dataset_name)
            # save_root = '/datagrid/personal/neoral/tmp/raft_export/sintel/{:s}_{:s}'.format(dataset_name, dataset_subname)

            ITERS = args.iters

            model = torch.nn.DataParallel(RAFT(args))
            model.load_state_dict(torch.load(args.model))

            model = model.module
            model.to(DEVICE)
            model.eval()

            model_e = model

            for t_scale_i in range(args.time_scale):
                t_scale = t_scale_i + 1
                dir_list = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root,d))]
                dir_list.sort()
                pbar = tqdm(dir_list)
                for sequence in pbar:
                    for image_n in range(51):
                        path_im1 = os.path.join(data_root, '{:s}/frame_{:04d}.png'.format(sequence, image_n))
                        path_im2 = os.path.join(data_root, '{:s}/frame_{:04d}.png'.format(sequence, image_n + t_scale))

                        if not os.path.exists(path_im1) or not os.path.exists(path_im2):
                            continue

                        pbar.set_description('t_scale = {:d}: {:s}/{:02d} and {:02d}'.format(t_scale, sequence, image_n, image_n + t_scale))

                        with torch.no_grad():
                            # kitti images

                            image1, image1_orig = load_image(path_im1)
                            image2, image2_orig = load_image(path_im2)

                            padder = InputPadder(image1.shape, mode='sintel')
                            image1, image2 = padder.pad(image1.cuda(), image2.cuda())

                            _, flow_pr = model_e(image1, image2, iters=ITERS, test_mode=True)
                            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

                            flow_gen.save_outputs(image1_orig, image2_orig, flow,
                                                  os.path.join(save_root, 'time_scale_{:d}'.format(t_scale), 'forward'),
                                                  '{:s}/frame_{:04d}.png'.format(sequence, image_n))

                            if args.backward:
                                _, flow_pr = model_e(image2, image1, iters=ITERS, test_mode=True)
                                flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()
                                flow_gen.save_outputs(image2_orig, image1_orig, flow, os.path.join(save_root, 'time_scale_{:d}'.format(t_scale), 'backward'), '{:s}/frame_{:04d}.png'.format(sequence, image_n + t_scale))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--iters', type=int, default=16)
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--time_scale', type=int, default=5)
    parser.add_argument('--backward', action='store_true', help='compute backward flow')

    args = parser.parse_args()
    gen(args)