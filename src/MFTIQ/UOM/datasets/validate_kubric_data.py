import argparse
import numpy as np
import torch
from pathlib import Path
import os
from rich.progress import track
from rich import print
import time
import MFTIQ.UOM.utils.plotting as dp
from MFTIQ.utils.io import read_flowou


def is_valid_file(filepath, min_non_occluded=None):
    flow, occl, unc = read_flowou(filepath)
    if flow is not None:
        if not np.all(np.isfinite(flow)):
            return False
    if occl is not None:
        if not np.all(np.isfinite(occl)):
            return False
        if min_non_occluded is not None:
            # check if at least "min_non_occluded" pixels is non-occluded
            if occl.size * min_non_occluded > np.sum(occl < 0.5):
                return False
    if unc is not None:
        if not np.all(np.isfinite(unc)):
            return False

    return True


def is_valid(dirpath, args):
    validity = False
    for c_file in dirpath.iterdir():
        filename = c_file.name
        if 'rgb' in filename:
            validity = True
        elif 'flow_gt' in filename:
            validity = is_valid_file(c_file, min_non_occluded=args.min_non_occluded)
        elif 'flow_est' in filename:
            validity = is_valid_file(c_file)
        else:
            validity = True
            # validity = is_valid_file(c_file)
        if not validity:
            break
    return validity


def move_dir(dirpath, args):
    args.thresh_path.mkdir(parents=True, exist_ok=True)
    dirpath.rename(args.thresh_path / dirpath.name)

def main(args):
    n_dirs = len(list(args.data_root.iterdir()))
    valid_dirs = 0
    invalid_dirs = 0
    for c_dir in track(sorted(args.data_root.iterdir()), description="Checking Kubric dir:", total=n_dirs):

        if is_valid(c_dir, args):
            valid_dirs += 1
            sstring = '[green]   valid'
        else:
            invalid_dirs += 1
            sstring = '[red] invalid'
            move_dir(c_dir, args)
        print(f'[green] {valid_dirs}, [red] {invalid_dirs} [blue] | {sstring}  -  {c_dir.name}')


def parse_arguments():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_root', type=Path, default=Path('/datagrid/tlabdata/neoramic/mft2024/kubric_datasets/20240310/'))
    parser.add_argument('--thresh_path', type=Path, default=Path('/datagrid/tlabdata/neoramic/mft2024/kubric_datasets/20240310_thresh/'))
    parser.add_argument('--min_non_occluded', type=float, default=0.25)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    # if args.gpuid is not None:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpuid}'
    # test(args)
    main(args)