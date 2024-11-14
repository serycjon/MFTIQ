import sys

sys.path.append("/datagrid/personal/neoral/repos/raft_new_debug")
sys.path.append('core')

import os
import numpy as np
from PIL import Image

from RAFT.core.utils import flow_viz
import RAFT.core.utils.flow_gen as flow_gen
from tqdm import tqdm

DEVICE = 'cuda'
MAX_FLOW_VIS = 20

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    return img



def gen():

    data_root = '/datagrid/tlab/data/MoSegUnexpected/images_halfres'
    all_dirs = ['/datagrid/tlab/data/MoSegUnexpected/flow_halfres/raft/model-sintel',
                '/datagrid/tlab/data/MoSegUnexpected/flow_halfres/raft/model-kitti']

    for save_root in all_dirs:

        for t_scale_i in range(5):
            t_scale = t_scale_i + 1

            subdirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root,d))]
            pbar = tqdm(subdirs)
            for sequence in pbar:

                cur_dir = os.path.join(data_root, sequence)

                files_list = [d for d in os.listdir(cur_dir) if os.path.isfile(os.path.join(cur_dir, d))]
                files_list.sort()

                n_files = len(files_list)



                for image_n in range(n_files - t_scale):

                    path_im1 = os.path.join(cur_dir, files_list[image_n])
                    try:
                        path_im2 = os.path.join(cur_dir, files_list[image_n + t_scale])
                    except:
                        continue

                    dir_flow_fw = os.path.join(save_root, sequence, 'time_scale_{:d}'.format(t_scale), 'forward')
                    filename_flow_fw = files_list[image_n]

                    dir_flow_bw = os.path.join(save_root, sequence, 'time_scale_{:d}'.format(t_scale), 'backward')
                    filename_flow_bw = files_list[image_n + t_scale]

                    if not os.path.exists(os.path.join(dir_flow_fw, 'flow', filename_flow_fw)) or not os.path.exists(os.path.join(dir_flow_bw, 'flow', filename_flow_bw)):
                        continue

                    if not os.path.exists(path_im1) or not os.path.exists(path_im2):
                        continue

                    image1 = load_image(path_im1)
                    image2 = load_image(path_im2)

                    flow_fw = flow_gen.read_flow_png_python(os.path.join(dir_flow_fw, 'flow', filename_flow_fw))
                    flow_bw = flow_gen.read_flow_png_python(os.path.join(dir_flow_bw, 'flow', filename_flow_bw))

                    flow_fw_image = flow_viz.flow_to_image(flow_fw, clip_flow=MAX_FLOW_VIS)
                    flow_bw_image = flow_viz.flow_to_image(flow_bw, clip_flow=MAX_FLOW_VIS)

                    flow_gen.save_image(np.concatenate([flow_fw_image, image1], axis=0), os.path.join(dir_flow_fw, 'flow_color', filename_flow_fw))
                    flow_gen.save_image(np.concatenate([flow_bw_image, image2], axis=0), os.path.join(dir_flow_bw, 'flow_color', filename_flow_bw))


if __name__ == '__main__':
    gen()