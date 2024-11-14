
import torch
from sklearn.decomposition import PCA
import einops
import numpy as np
import os
import matplotlib
from MFTIQ.RAFT.core.utils import flow_viz
import cv2
from PIL import Image


if os.getenv('REMOTE_DEBUG'):
    matplotlib.use('module://backend_interagg')
print(f'{matplotlib.get_backend() = }')

import matplotlib.pyplot as plt

def get_pca(x, n_components=3, normalized=True, **kwargs):
    """Plot PCA color image of input

    Args:
        x: image data [C, H, W]

    Returns:
    """

    pca = PCA(n_components=n_components)
    x_np = x.detach().cpu().numpy()
    x_shape = einops.parse_shape(x, 'C H W') 
    x_np = einops.rearrange(x_np, 'C H W -> (H W) C')
    reduced_x_flat = pca.fit_transform(x_np.astype(np.float32))
    reduced_x = einops.rearrange(reduced_x_flat, '(H W) C -> H W C', H=x_shape['H'], W=x_shape['W'], C=n_components)
    if normalized:
        return (reduced_x - np.min(reduced_x)) / (np.max(reduced_x) - np.min(reduced_x))
    return reduced_x

def plot_pca(x, normalized=True, **kwargs):
    x_norm = get_pca(x, normalized=normalized, **kwargs)
    plot_torch_img(x_norm, multiply_255=True, pattern='H W C -> H W C')

def plot_torch_img(img, multiply_255=False, pattern=None):
    if img.dtype in [torch.float32, torch.uint8, torch.float16]:
        img = img.detach().cpu().numpy()
    if pattern is None:
        pattern = 'C H W -> H W C'
    img_np = einops.rearrange(img, pattern)
    if multiply_255:
        img_np = img_np * 255
    plt.imshow(np.clip(img_np, 0, 255).astype(np.uint8))
    plt.show()


def save_multi_image(img1, img2, flow, flow_init, flow_noinit, idx1=None, idx2=None, delta=None, savedir=None):
    if savedir is None:
        print('something wrong')
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1.5
    color = (255, 255, 0)
    thickness = 2

    img1t = cv2.putText(img1.copy(), f'img {idx1}', org, font, fontScale, color, thickness, cv2.LINE_AA)
    img2t = cv2.putText(img2.copy(), f'img {idx2}', org, font, fontScale, color, thickness, cv2.LINE_AA)
    img1t = cv2.cvtColor(img1t, cv2.COLOR_BGR2RGB)
    img2t = cv2.cvtColor(img2t, cv2.COLOR_BGR2RGB)

    flow_diff = flow.flow - flow_noinit.flow
    flow_cat = torch.cat([flow.flow, flow_init, flow_noinit.flow, flow_diff], dim=-1)
    flow_cat_np = flow_cat.permute(1,2,0).cpu().numpy()
    flow_cat_img = flow_viz.flow_to_image(flow_cat_np, clip_flow=150)

    flow_img_split = np.split(flow_cat_img, 4, axis=1)
    flow_t = cv2.putText(flow_img_split[0], f'flow delta={delta} with init', org, font, fontScale, color, thickness, cv2.LINE_AA)
    flow_init_t = cv2.putText(flow_img_split[1], f'flow init', org, font, fontScale, color, thickness, cv2.LINE_AA)
    flow_noinit_t = cv2.putText(flow_img_split[2], f'flow delta={delta} noinit', org, font, fontScale, color, thickness, cv2.LINE_AA)
    flow_diff_t = cv2.putText(flow_img_split[3], f'flow diff', org, font, fontScale, color, thickness, cv2.LINE_AA)

    imgv = np.concatenate([img1t, img2t], axis=0)
    flow1v = np.concatenate([flow_t, flow_init_t], axis=0)
    flow2v = np.concatenate([flow_noinit_t, flow_diff_t], axis=0)
    final_img = np.concatenate([imgv,flow1v,flow2v], axis=1)

    savedir = savedir / f'{delta}'
    savedir.mkdir(exist_ok=True, parents=True)
    im = Image.fromarray(final_img)
    im.save(f"{str(savedir)}/{idx1:03d}_to_{idx2:03d}.jpg")





