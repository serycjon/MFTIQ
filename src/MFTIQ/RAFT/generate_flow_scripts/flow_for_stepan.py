import sys
sys.path.append("/datagrid/personal/neoral/repos/raft_debug")
sys.path.append('core')

import argparse
import os
import torch

from RAFT.core import RAFT
import RAFT.core.utils.flow_gen as flow_gen
from tqdm import tqdm

DEVICE = 'cuda'

def gen(args):

    CHECKPOINT = {
        # 'sintel': "models/sintel.pth",
        'kitti': "models/kitti.pth",
        # 'things3d': "models/chairs+things.pth"
    }

    for dataset_name, checkpoint_path in CHECKPOINT.items():
        args.model = checkpoint_path

        data_root = '/datagrid/personal/sochman/tmp/EUTour_bits/letna_left_wide_imgs'
        save_root = '/datagrid/personal/neoral/datasets/EUTour_bits/letna_left_wide_imgs/raft_export/{:s}'.format(dataset_name)

        model = RAFT(args)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(args.model))

        model.to(DEVICE)
        model.eval()

        for t_scale_i in range(1):
            t_scale = t_scale_i + 1
            # files = os.listdir(data_root)[0:2]
            # files = [f for f in files if files if f.endswith('.png')]
            pbar = tqdm(range(4600))
            for image_n in pbar:
                path_im1 = os.path.join(data_root, 'frame_{:05d}.png'.format(image_n))
                path_im2 = os.path.join(data_root, 'frame_{:05d}.png'.format(image_n + t_scale))

                if not os.path.exists(path_im1) or not os.path.exists(path_im2):
                    continue

                pbar.set_description('t_scale = {:d}: {:05d}'.format(t_scale, image_n))

                with torch.no_grad():
                        # kitti images
                        image1 = flow_gen.load_image(path_im1)
                        image2 = flow_gen.load_image(path_im2)
                        flow_predictions = model.module(image1, image2, iters=16)
                        flow_gen.save_outputs(image1[0], image2[0], flow_predictions[-1][0], os.path.join(save_root, 'time_scale_{:d}'.format(t_scale), 'forward'), 'frame_{:05d}.png'.format(image_n), clip_flow=30)

                        # flow_predictions = model.module(image2, image1, iters=16)
                        # flow_gen.save_outputs(image2[0], image1[0], flow_predictions[-1][0], os.path.join(save_root, 'time_scale_{:d}'.format(t_scale), 'backward'), 'frame_{:05d}.png'.format(image_n + t_scale))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint", default='')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--iters', type=int, default=12)

    args = parser.parse_args()
    gen(args)