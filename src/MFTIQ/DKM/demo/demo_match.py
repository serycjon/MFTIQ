from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from ipdb import iex

from MFTIQ.DKM.dkm.utils.utils import tensor_to_pil
from MFTIQ.DKM.dkm import DKMv3_outdoor
from MFTIQ.utils import vis_utils as vu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pil_to_gray_tensor(im):
    if not isinstance(im, torch.Tensor):
        im = np.array(im)[:, :, ::-1] # RGB to BGR
        im = vu.to_gray_3ch(im)
        im = im.astype(np.float32).transpose((2, 0, 1))
        # im = np.array(im, dtype=np.float32).transpose((2, 0, 1))
        im /= 255.0
        return torch.from_numpy(im)
    else:
        return im

@iex
def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--im_A_path", default="assets/sacre_coeur_A.jpg", type=str)
    parser.add_argument("--im_B_path", default="assets/sacre_coeur_B.jpg", type=str)
    parser.add_argument("--save_path", default="demo/dkmv3_warp_sacre_coeur.jpg", type=str)

    args, _ = parser.parse_known_args()
    im1_path = args.im_A_path
    im2_path = args.im_B_path
    save_path = args.save_path

    # Create model
    dkm_model = DKMv3_outdoor(device=device)

    H, W = 864, 1152

    im1 = Image.open(im1_path).resize((W, H))
    im2 = Image.open(im2_path).resize((W, H))

    # Match
    warp, certainty = dkm_model.match(im1_path, im2_path, device=device)
    # Sampling not needed, but can be done with model.sample(warp, certainty)
    dkm_model.sample(warp, certainty)
    x1 = (torch.tensor(np.array(im1)) / 255).to(device).permute(2, 0, 1)
    x2 = (torch.tensor(np.array(im2)) / 255).to(device).permute(2, 0, 1)

    cb = vu.color_checkerboard(H, W, 40)[:, :, :3][:, :, ::-1].copy()  # BGRA to RGB
    torch_cb = (torch.tensor(cb) / 255).to(device).permute(2, 0, 1)

    im2_transfer_rgb = F.grid_sample(
    x2[None], warp[:,:W, 2:][None], mode="bilinear", align_corners=False
    )[0]
    im1_transfer_rgb = F.grid_sample(
    x1[None], warp[:, W:, :2][None], mode="bilinear", align_corners=False
    )[0]
    cb_1_transfer_rgb = F.grid_sample(
    torch_cb[None], warp[:,:W, 2:][None], mode="bilinear", align_corners=False
    )[0]
    cb_2_transfer_rgb = F.grid_sample(
    torch_cb[None], warp[:, W:, :2][None], mode="bilinear", align_corners=False
    )[0]
    warp_im = torch.cat((im2_transfer_rgb,im1_transfer_rgb),dim=2)
    white_im = torch.ones((H,2*W),device=device)
    white_im = torch.cat([pil_to_gray_tensor(im1), pil_to_gray_tensor(im2)], dim=2).to(device)
    vis_im = certainty * warp_im + (1 - certainty) * white_im
    cb_im = torch.cat((cb_1_transfer_rgb, cb_2_transfer_rgb), dim=2)
    vis_im = torch.cat((vis_im, cb_im), dim=1)
    tensor_to_pil(vis_im, unnormalize=False).save(save_path)

if __name__ == "__main__":
    main()
