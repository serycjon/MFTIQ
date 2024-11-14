from __future__ import print_function, division
import sys, os
from RAFT.core.raft import RAFT
import tqdm
import os
import numpy as np

import torch
from RAFT.core.utils.utils import InputPadder

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time


os.chdir('/mnt/datagrid/personal/neoral/repos/raft_new_debug/RAFT/')
#%%
myhost = os.uname()[1]
print(myhost)
#%%
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

class cuda_time_measurer():
    """ https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964/2
    https://auro-227.medium.com/timing-your-pytorch-code-fragments-e1a556e81f2 """

    def __init__(self, units=None):
        # self.start_time = timer()
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

        self.units = units

        self.start_event.record()

    def __call__(self):
        self.end_event.record()
        torch.cuda.synchronize()  # this is intentionally and correctly here, AFTER the end_event.record()
        value = self.start_event.elapsed_time(self.end_event)
        return value

#%%
default_kwargs = {
    'model': '/home.stud/neoramic/repos/raft_new_debug/RAFT/checkpoints/raft-things-sintel-kubric-splitted-occlusion-uncertainty-non-occluded-base-sintel.pth',
    'save_path': '/datagrid/personal/neoral/RAFT_occl_uncertainty_outputs/statistics_occl',
    'iters': 12,
    'gpus': 0,
    'debug': True,
    'subsplit': 'validation'
}
args = AttrDict(**default_kwargs)
#%%
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
raft.eval();
#%%
B = 1
R_list = list(range(64, 1600, 8))

values = []

for R in R_list:
    im1_np = 2 * np.random.random((B, 3, R, 2*R)).astype(np.float32) - 1.0
    im2_np = 2 * np.random.random((B, 3, R, 2*R)).astype(np.float32) - 1.0

    im1 = torch.from_numpy(im1_np)
    im2 = torch.from_numpy(im2_np)

    padder = InputPadder(im1.shape, mode='sintel')
    im1, im2 = padder.pad(im1, im2)

    im1 = im1.to(device=device)
    im2 = im2.to(device=device)

    #%%
    number_of_comp = 10

    # warm up
    try:
        for i in range(3):
            all_predictions = raft(im1.to(device), im2.to(device), iters=args.iters, test_mode=True)


        timer_list = []
        for i in range(number_of_comp):
            timer = cuda_time_measurer()
            all_predictions = raft(im1.to(device), im2.to(device), iters=args.iters, test_mode=True)
            timer_list.append(timer())
        values.append(np.mean(timer_list))
        print(f'Mean elapsed time {np.mean(timer_list)}ms, median time {np.median(timer_list)}, resolution WH {2*R}x{R}')
    except:
        values.append(np.nan)

plt.figure(figsize=(10,5))
plt.plot(R_list, values)
plt.ylabel('speed [ms]')
plt.xlabel('size [px]')
plt.title('RAFT speed on size')
plt.grid(which='major', color='#CCCCCC', linewidth=0.9)
plt.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.7)
plt.minorticks_on()
plt.tight_layout()
plt.savefig('speed_measurements_WtoH_2to1_sintel.png', dpi=300)
# plt.show()


plt.figure(figsize=(10,5))
plt.plot(np.array(R_list)*2*np.array(R_list), values)
plt.title('RAFT speed on resolutions')
plt.ylabel('speed [ms]')
plt.xlabel('resolution [px$^2$]')
plt.grid(which='major', color='#CCCCCC', linewidth=0.9)
plt.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.7)
plt.minorticks_on()
plt.tight_layout()
plt.savefig('speed_measurements_resolutions_WtoH_2to1_sintel.png', dpi=300)
# plt.show()
