import sys

sys.path.append("/datagrid/personal/neoral/repos/raft_new_debug")
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
MAX_FLOW_VIS = 20

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img_torch = torch.from_numpy(img).permute(2, 0, 1).float()
    return img_torch[None].to(DEVICE), img


@torch.no_grad()
def gen(args):

    if 'kitti' in args.model:
        short_model_name = 'kitti'
    elif 'sintel' in args.model:
        short_model_name = 'sintel'
    elif 'things' in args.model:
        short_model_name = 'things'
    else:
        short_model_name = args.model[:-4]

    if args.fullres:
        data_root = '/datagrid/tlab/data/MoSegUnexpected/images'
        save_root = '/datagrid/tlab/data/MoSegUnexpected/flow_fullres/raft/model-{:s}'.format(short_model_name)
    else:
        data_root = '/datagrid/tlab/data/MoSegUnexpected/images_halfres'
        save_root = '/datagrid/tlab/data/MoSegUnexpected/flow_halfres/raft/model-{:s}'.format(short_model_name)

    print('data_root: ' + data_root)
    print('save_root: ' + save_root)

    ITERS = 24

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    model_e = model

    for t_scale_i in range(args.time_scale):
        t_scale = t_scale_i + 1

        subdirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root,d))]
        pbar = tqdm(subdirs)
        for sequence in pbar:

            cur_dir = os.path.join(data_root, sequence)

            files_list = [d for d in os.listdir(cur_dir) if os.path.isfile(os.path.join(cur_dir, d))]
            files_list.sort()

            n_files = len(files_list)



            for image_n in range(args.min_frame, min(args.max_frame,n_files - t_scale)):

                path_im1 = os.path.join(cur_dir, files_list[image_n])
                try:
                    path_im2 = os.path.join(cur_dir, files_list[image_n + t_scale])
                except:
                    continue

                dir_flow_fw = os.path.join(save_root, sequence, 'time_scale_{:d}'.format(t_scale), 'forward')
                filename_flow_fw = files_list[image_n]

                dir_flow_bw = os.path.join(save_root, sequence, 'time_scale_{:d}'.format(t_scale), 'backward')
                filename_flow_bw = files_list[image_n + t_scale]

                if os.path.exists(os.path.join(dir_flow_fw, 'flow', filename_flow_fw)) and os.path.exists(os.path.join(dir_flow_bw, 'flow', filename_flow_bw)):
                    continue

                if not os.path.exists(path_im1) or not os.path.exists(path_im2):
                    continue

                with torch.no_grad():
                    # kitti images

                    try:
                        image1, image1_orig = load_image(path_im1)
                        image2, image2_orig = load_image(path_im2)
                    except:
                        print('cannot load im1: '+path_im1+ ' or im2: ' + path_im2)
                        continue

                    padder = InputPadder(image1.shape, mode='kitti')
                    image1, image2 = padder.pad(image1.cuda(), image2.cuda())

                    _, flow_pr = model_e(image1, image2, iters=ITERS, test_mode=True)
                    flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

                    # output_filename = os.path.join(save_root, 'time_scale_{:d}'.format(t_scale), 'forward'), '{:06d}_{:02d}.png'.format(sequence, image_n)
                    # frame_utils.writeFlowKITTI(output_filename, flow)

                    # flow_predictions = model(image1, image2, iters=16, test_mode=True)
                    flow_gen.save_outputs(image1_orig, image2_orig, flow, dir_flow_fw, filename_flow_fw, clip_flow=MAX_FLOW_VIS)

                    if args.backward:
                        # flow_predictions = model(image2, image1, iters=16, test_mode=True)
                        _, flow_pr = model_e(image2, image1, iters=ITERS, test_mode=True)
                        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()
                        flow_gen.save_outputs(image2_orig, image1_orig, flow, dir_flow_bw, filename_flow_bw, clip_flow=MAX_FLOW_VIS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--fullres', action='store_true', help='use small model')
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--gpu', help="number of CUDA_VISIBLE_DEVICES", default='0')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    parser.add_argument('--time_scale', type=int, default=5)
    parser.add_argument('--min_frame', type=int, default=0)
    parser.add_argument('--max_frame', type=int, default=1000)
    parser.add_argument('--backward', action='store_true', help='compute backward flow')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    gen(args)