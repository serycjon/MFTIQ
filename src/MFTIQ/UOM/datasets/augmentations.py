# MFTIQ - WACV2025
import numpy as np
import einops
import torch
import MFTIQ.UOM.utils.image_manipulation as im

# import MFT.UOM.utils.plotting as dp

class Rotate90Augmentor:
    def __init__(self, enabled):
        self.enabled = enabled

        self.rotate_clockwise = np.random.choice([True, False])

    def __call__(self, data):
        if not self.enabled:
            return data

        # data['flow'] = data['flow'] * 0.0
        # data = {k:v[:,:,:368,:368] for k, v in data.items()}

        data_orig = {k:v for k,v in data.items()}
        for k, v in data.items():
            if 'flow_speed' in k or 'flow_position' in k:
                raise NotImplementedError(f'Rotate90 is not implemented for {k} YET!')
            elif 'flow' in k:
                data[k] = self.flow_transform(v)
            elif 'img2' in k or 'image2' in k:
                data[k] = self.image_transform(v)

        # x1_shape = einops.parse_shape(data['img1'], 'B C H W')
        # x2_shape = einops.parse_shape(data['img2'], 'B C H W')
        #
        # im1_orig, im2_orig, _ = im.universal_warp_backward(data_orig['img1'], data_orig['img2'], data_orig['flow'])
        # im1, im2, _ = im.universal_warp_backward(data['img1'], data['img2'], data['flow'],
        #                                          x1_orig_shape=x1_shape, x2_orig_shape=x2_shape)
        # dp.plot_torch_img(im2_orig[4])
        # dp.plot_torch_img(im2[4])

        return data

    def image_transform(self, img):
        if self.rotate_clockwise:
            img = torch.rot90(img, -1, (2, 3))
        else:
            img = torch.rot90(img, 1, (2,3))
        return img

    def flow_transform(self, flow):
        coords1, coords2 = im.flow_to_coords(einops.rearrange(flow, 'B C H W -> B H W C'))
        coor2_x, coor2_y = coords2.split(1, dim=-1)
        shape_parsed = einops.parse_shape(flow, 'B C H W')

        if self.rotate_clockwise:
            coords2_rot = torch.cat([(shape_parsed['H'] - 1) - coor2_y, coor2_x], dim=-1)
        else:
            coords2_rot = torch.cat([coor2_y, (shape_parsed['W'] - 1) - coor2_x], dim=-1)

        coords_diff = coords2_rot - coords1
        flow = einops.rearrange(coords_diff, 'B H W C -> B C H W')

        return flow