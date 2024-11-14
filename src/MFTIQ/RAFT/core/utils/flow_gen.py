import sys
sys.path.append('core')

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import imageio
import png

from MFTIQ.RAFT.core.utils import flow_viz

DEVICE = 'cuda'

def pad8(img):
    """pad image such that dimensions are divisible by 8"""
    ht, wd = img.shape[2:]
    pad_ht = (((ht // 8) + 1) * 8 - ht) % 8
    pad_wd = (((wd // 8) + 1) * 8 - wd) % 8
    pad_ht1 = [pad_ht//2, pad_ht-pad_ht//2]
    pad_wd1 = [pad_wd//2, pad_wd-pad_wd//2]

    img = F.pad(img, pad_wd1 + pad_ht1, mode='replicate')
    return img


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)[..., :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return pad8(img[None]).to(DEVICE)


def display(image1, image2, flow):
    image1 = image1.permute(1, 2, 0).cpu().numpy() / 255.0
    image2 = image2.permute(1, 2, 0).cpu().numpy() / 255.0

    flow = flow.permute(1, 2, 0).cpu().numpy()
    flow_image = flow_viz.flow_to_image(flow)
    flow_image = cv2.resize(flow_image, (image1.shape[1], image1.shape[0]))


    cv2.imshow('image1', image1[..., ::-1])
    cv2.imshow('image2', image2[..., ::-1])
    cv2.imshow('flow', flow_image[..., ::-1])
    cv2.waitKey()


def create_dir_if_no_exists(path):
    splitted = path.split('/')
    dir_path = os.path.join('/', *splitted[:-1])
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def save_image(im, path):
    create_dir_if_no_exists(path)
    imageio.imwrite(path, im)


def save_flow_PNG(flow, path):
    flow = flow.astype(np.double)
    flow[:, :, 0] = np.clip(flow[:, :, 0] * 64. + 2 ** 15, 0, 2 ** 16 - 1)
    flow[:, :, 1] = np.clip(flow[:, :, 1] * 64. + 2 ** 15, 0, 2 ** 16 - 1)
    if flow.shape[2] == 3:
        flow[:, :, 2] = np.clip(flow[:, :, 2], 0., 1.)
    else:
        ones = np.ones([flow.shape[0], flow.shape[1], 1], np.float32)
        flow = np.concatenate([flow, ones], axis=2)
    flow = flow.astype(np.uint16)
    save_png(flow, path, bitdepth=16)


def save_png(img, filename, bitdepth=8, grayscale=False):
    splitted = filename.split('/')
    dir_path = os.path.join('/', *splitted[:-1])
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(filename, 'wb') as f:
        writer = png.Writer(width=img.shape[1], height=img.shape[0], bitdepth=bitdepth, greyscale=grayscale)
        if grayscale:
            z2list = img.tolist()
        else:
            z2list = img.reshape(-1, img.shape[1] * img.shape[2]).tolist()
        writer.write(f, z2list)


def read_png_python(filepath, dtype=None, channels=1):
    dtype = dtype if dtype is not None else np.uint8
    reader = png.Reader(filepath)
    pngdata = reader.read()
    px_array = np.array(list(map(dtype, pngdata[2])))
    if channels != 1:
        px_array = px_array.reshape(-1, np.int32(px_array.shape[1] // channels), channels)
        if channels == 4:
            px_array = px_array[:,:,0:3]
    else:
        px_array = np.expand_dims(px_array, axis=2)
    return px_array


def read_flow_png_python(file_path):
    flow = read_png_python(file_path, channels=3, dtype=np.uint16)
    flow = flow.astype(np.float32)
    u, v, valid = flow[:,:,0], flow[:,:,1], flow[:,:,2]
    u = (u - 2 ** 15) / 64.
    v = (v - 2 ** 15) / 64.
    flow = np.stack([u, v, valid], axis=2)
    return flow


def save_outputs(image1, image2, flow, path, filename, clip_flow=None):
    # image1 = image1.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    # image2 = image2.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    #
    # flow = flow.permute(1, 2, 0).cpu().numpy()
    flow_image = flow_viz.flow_to_image(flow, clip_flow=clip_flow)
    flow_image = cv2.resize(flow_image, (image1.shape[1], image1.shape[0]))

    save_image(np.concatenate([flow_image, image1], axis=0), os.path.join(path, 'flow_color', filename[:-3]+"jpg"))
    save_flow_PNG(flow, os.path.join(path, 'flow', filename))

