# -*- origami-fold-style: triple-braces -*-
from __future__ import print_function, division
import sys
from ipdb import iex

import argparse
import numpy as np
import random
import einops

import cv2
import h5py
import tqdm

import torch
import RAFT.core.datasets as datasets


@iex
def prep_dataset(args):
    args.n_workers = 0
    train_loader = datasets.fetch_dataloader(args)
    left_images = []
    right_images = []
    flows = []
    valid_masks = []
    occlusion_maps = []
    for i_batch, data_blob in tqdm.tqdm(enumerate(train_loader), total=args.num_samples):
        if i_batch >= args.num_samples:
            break
        image1, image2, flow, valid, occl = [x.cpu().numpy() for x in data_blob]
        left_images.append(image1)
        right_images.append(image2)
        flows.append(flow)
        valid_masks.append(valid)
        occlusion_maps.append(occl)
        # all_predictions = model(image1, image2, iters=args.iters)

        if args.debug:  # visualization {{{
            cv2.namedWindow("cv: img1", cv2.WINDOW_NORMAL)
            img1 = einops.rearrange(image1.cpu().numpy(), 'B C H W -> H (B W) C', C=3)[:, :, ::-1]
            img1 = np.round(img1).astype(np.uint8)

            cv2.namedWindow("cv: img2", cv2.WINDOW_NORMAL)
            img2 = einops.rearrange(image2.cpu().numpy(), 'B C H W -> H (B W) C', C=3)[:, :, ::-1]
            img2 = np.round(img2).astype(np.uint8)

            cv2.imshow("cv: img1", img1)
            cv2.imshow("cv: img2", img2)
            while True:
                c = cv2.waitKey(0)
                if c == ord('q'):
                    sys.exit(1)
                elif c == ord(' '):
                    break
                # }}}

    with h5py.File('pokus.hdf5', 'w') as f:
        common_args = dict(compression='gzip', shuffle=True)
        if False:
            zipped_data = np.array([np.array(*xs) for xs in zip(left_images, right_images, flows, valid_masks, occlusion_maps)])
            f.create_dataset("dataset", data=zipped_data, **common_args)
        else:
            f.create_dataset("left_images", data=left_images, **common_args)
            f.create_dataset("right_images", data=right_images, **common_args)
            f.create_dataset("flows", data=flows, **common_args)
            f.create_dataset("valid_masks", data=valid_masks, **common_args)
            f.create_dataset("occlusion_maps", data=occlusion_maps, **common_args)

    return 0


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RAFT PyTorch implementation.', fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--num_samples', help='', type=int, default=100)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])

    parser.add_argument('--dashcam_augmenentation', action='store_true')
    parser.add_argument('--blend_source', default='/datagrid/public_datasets/COCO/train2017',
                        help="path to blending images")
    parser.add_argument('--seed', help='', type=int, default=1234)
    parser.add_argument('--debug', help='', action='store_true')

    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = f'@{sys.argv[1]}'
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.occlusion_module = True

    prep_dataset(args)
