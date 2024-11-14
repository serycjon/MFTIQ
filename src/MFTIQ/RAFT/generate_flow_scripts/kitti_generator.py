import sys
sys.path.append("/datagrid/personal/neoral/repos/raft_debug")
sys.path.append('core')

import argparse
import os
import numpy as np
import torch
from PIL import Image

from RAFT.core.raft import RAFT
import RAFT.core.utils.flow_gen as flow_gen
from tqdm import tqdm

from RAFT.core.utils.utils import InputPadder

DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img_torch = torch.from_numpy(img).permute(2, 0, 1).float()
    return img_torch[None].to(DEVICE), img

@torch.no_grad()
def gen(args):

    for dataset_name in ['testing', 'training']:

        if 'kitti' in args.model:
            short_model_name = 'kitti'
        elif 'sintel' in args.model:
            short_model_name = 'sintel'
        elif 'things' in args.model:
            short_model_name = 'things'
        else:
            short_model_name = args.model[:-4]

        data_root = '/datagrid/public_datasets/KITTI/multiview/{:s}/image_2'.format(dataset_name)
        save_root = '/datagrid/personal/neoral/datasets/optical_flow_neomoseg/raft_new_export/kitti_{:s}_model/{:s}'.format(short_model_name, dataset_name)
        # save_root = '/datagrid/tlab/personal/neoramic/datasets/optical_flow_neomoseg/raft_new_export/kitti_aug_model/{:s}'.format(dataset_name)

        ITERS = args.iters

        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(args.model))

        model = model.module
        model.to(DEVICE)
        model.eval()

        model_e = model
    
        for t_scale_i in range(args.time_scale):
            t_scale = t_scale_i + 1
            pbar = tqdm(range(200))
            # pbar = tqdm([11,122,15,177])
            for sequence in pbar:
                for image_n in range(args.min_frame, args.max_frame + 1):
                    path_im1 = os.path.join(data_root, '{:06d}_{:02d}.png'.format(sequence, image_n))
                    path_im2 = os.path.join(data_root, '{:06d}_{:02d}.png'.format(sequence, image_n + t_scale))
                    
                    if not os.path.exists(path_im1) or not os.path.exists(path_im2):
                        continue
                    
                    pbar.set_description('t_scale = {:d}: {:06d}: {:02d} and {:02d}'.format(t_scale, sequence, image_n, image_n + t_scale))
    
                    with torch.no_grad():
                        # kitti images
                        image1, image1_orig = load_image(path_im1)
                        image2, image2_orig = load_image(path_im2)

                        padder = InputPadder(image1.shape, mode='kitti')
                        image1, image2 = padder.pad(image1.cuda(), image2.cuda())

                        _, flow_pr = model_e(image1, image2, iters=ITERS, test_mode=True)
                        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

                        #output_filename = os.path.join(save_root, 'time_scale_{:d}'.format(t_scale), 'forward'), '{:06d}_{:02d}.png'.format(sequence, image_n)
                        #frame_utils.writeFlowKITTI(output_filename, flow)

                        #flow_predictions = model(image1, image2, iters=16, test_mode=True)
                        flow_gen.save_outputs(image1_orig, image2_orig, flow, os.path.join(save_root, 'time_scale_{:d}'.format(t_scale), 'forward'), '{:06d}_{:02d}.png'.format(sequence, image_n))

                        if args.backward:
                            #flow_predictions = model(image2, image1, iters=16, test_mode=True)
                            _, flow_pr = model_e(image2, image1, iters=ITERS, test_mode=True)
                            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()
                            flow_gen.save_outputs(image2_orig, image1_orig, flow, os.path.join(save_root, 'time_scale_{:d}'.format(t_scale), 'backward'), '{:06d}_{:02d}.png'.format(sequence, image_n + t_scale))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--iters', type=int, default=24)
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--time_scale', type=int, default=5)
    parser.add_argument('--min_frame', type=int, default=0)
    parser.add_argument('--max_frame', type=int, default=20)
    parser.add_argument('--backward', action='store_true', help='compute backward flow')



    args = parser.parse_args()
    gen(args)