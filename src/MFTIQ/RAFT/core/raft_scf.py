import torch.nn.functional

from MFTIQ.RAFT.core.raft_orig import RAFT_ORIG
from enum import Enum
import einops
import numpy as np

class BlendingMethod(Enum):
    FURTHER_FROM_EDGE = 'further_from_edge'
    RANDOM = 'random'
    AVERAGE = 'average'
    CONFIDENCE = 'confidence'

# RAFT with Scale, Crop,and Fusion
class RAFTSCF(RAFT_ORIG):
    def __init__(self, args):
        super(RAFTSCF, self).__init__(args)

        # if enabled=False -> works as ordinary RAFT/RAFTou
        self.scf_enabled = args.get('scf_enabled', False)

        # scf_scale for image upsize
        self.scf_scale = args.get('scf_scale', 2)

        # crop minimum overlap
        self.scf_min_overlap_percent = args.get('scf_min_overlap_percent', 25)

        # what to do in overlying parts
        self.scf_blending_method = args.get('scf_blending_method', BlendingMethod.FURTHER_FROM_EDGE)

        # experimentally tested - maximum is 1872x936, for safety it is set to 1600x800
        self.scf_max_run_WH_px = args.get('scf_max_run_WH_px', {'W': 1600, 'H':800})

        # set interpolation for scaling operation (torch)
        self.scf_interpolate_type = args.get('scf_interpolate_type', 'bilinear')

        self.scf_align_corners = args.get('scf_align_corners')

        # add coloured image in the output to see blending
        self.scf_debug = args.get('scf_debug', True)

    def calculate_start_crop_windows(self, img):
        def calculate_n_crop_windows_single_dim(bbox_length_px, img_length_px, min_overlap_percent):
            if bbox_length_px >= img_length_px:
                return [0]
            min_overlap_pix = bbox_length_px * (min_overlap_percent / 100)
            img_length_without_borders_px = (img_length_px - min_overlap_pix)
            bbox_length_without_borders_px = (bbox_length_px - min_overlap_pix)
            n_bbox_single_dim_float = img_length_without_borders_px / bbox_length_without_borders_px
            n_bbox_single_dim = int(np.ceil(n_bbox_single_dim_float))

            sum_overlap = n_bbox_single_dim * bbox_length_px - img_length_px
            pos_diff = int(np.ceil(sum_overlap / (n_bbox_single_dim - 1)))

            start_points = []
            for i in range(n_bbox_single_dim):
                if i == (n_bbox_single_dim - 1):
                    start_points.append(img_length_px - bbox_length_px)
                else:
                    start_points.append((bbox_length_px - pos_diff) * i)
            return start_points

        max_run_WH_px = self.scf_max_run_WH_px
        min_overlap_percent = self.scf_min_overlap_percent

        img_shape = einops.parse_shape(img, 'N C H W')
        start_crop = {}
        for dim in ['W', 'H']:
            start_crop[dim] = calculate_n_crop_windows_single_dim(max_run_WH_px[dim], img_shape[dim], min_overlap_percent)
        return start_crop

    def create_crops(self, img, start_crops):
        def crop_torch_img(img, top, left, height, width):
            # print(top, top + height, left, left + width)
            return img[:, :, top:(top + height), left:(left + width)]

        bbox_length_px = self.scf_max_run_WH_px
        crops = []
        for idx_w in range(len(start_crops['W'])):
            for idx_h in range(len(start_crops['H'])):
                crops.append(crop_torch_img(img, top=start_crops['H'][idx_h], left=start_crops['W'][idx_w],
                                            height=bbox_length_px['H'], width=bbox_length_px['W']))
        return crops

    def create_position_masks(self, img_shape, start_crops, device):
        bbox_length_px = self.scf_max_run_WH_px
        def mark_position_in_mask(mask, top, left, height, width):
            # print(top, top + height, left, left + width)
            mask[:, :, top:(top + height), left:(left + width)] = True
            return mask

        masks = []
        for idx_w in range(len(start_crops['W'])):
            for idx_h in range(len(start_crops['H'])):
                c_mask = torch.zeros([img_shape['N'], 1, img_shape['H'], img_shape['W']], dtype=torch.bool, device=device)
                c_mask = mark_position_in_mask(c_mask, top=start_crops['H'][idx_h], left=start_crops['W'][idx_w],
                                               height=bbox_length_px['H'], width=bbox_length_px['W'])
                masks.append(c_mask)
        return masks

    def create_overlap_map(self, position_masks):
        stack_pos_masks = torch.stack(position_masks, dim=0)
        return torch.sum(stack_pos_masks, dim=0, keepdim=False)

    def place_crops_to_correct_position(self, img_shape, crops, start_crops, device):
        bbox_length_px = self.scf_max_run_WH_px
        def place_crop(mask, crop, top, left, height, width):
            mask[:, :, top:(top + height), left:(left + width)] = crop
            return mask

        placed_crops = []
        idx_crop = 0
        for idx_w in range(len(start_crops['W'])):
            for idx_h in range(len(start_crops['H'])):
                c_crop = crops[idx_crop]
                crop_shape = einops.parse_shape(c_crop, 'N C H W')
                c_data = torch.zeros([img_shape['N'], crop_shape['C'], img_shape['H'], img_shape['W']], dtype=torch.float32, device=device)
                c_data = place_crop(c_data, c_crop, top=start_crops['H'][idx_h], left=start_crops['W'][idx_w],
                                    height=bbox_length_px['H'], width=bbox_length_px['W'])
                placed_crops.append(c_data)
                idx_crop += 1
        return placed_crops

    def blend_average(self, placed_crops, overlap_count):
        sum_image = torch.sum(torch.stack(placed_crops, dim=0), dim=0)
        return sum_image / overlap_count

    def generate_random_maps(self, positions_masks, overlap_count):
        random_maps = []
        overlap_mask = overlap_count > 1
        not_overlap_mask = overlap_count <= 1
        for pm in positions_masks:
            c_random_map = torch.rand_like(pm, dtype=torch.float32) * pm * overlap_mask
            c_random_map += (pm * not_overlap_mask).to(torch.float32)
            random_maps.append(c_random_map)
        return random_maps

    def blend_weight(self, placed_crops, weights, minimum=False):
        stacked_data = torch.stack(placed_crops, dim=0)
        stacked_weights = torch.stack(weights, dim=0)
        if minimum:
            stacked_weights = -stacked_weights
        argmax_weights = torch.argmax(stacked_weights, dim=0, keepdim=True)
        C = placed_crops[0].shape[1]
        argmax_weights = torch.cat([argmax_weights] * C, dim=2)
        return torch.gather(stacked_data, 0, argmax_weights)[0]

    def convert_sigmas_to_weights(self, img_shape, sigma_maps, start_crops, device):
        sigma_maps = [1000000. - sm for sm in sigma_maps]
        return self.place_crops_to_correct_position(img_shape, sigma_maps, start_crops, device=device)


    def compute_distance_from_edge(self, crops):
        distance_maps = []
        for c in crops:
            crop_shape = einops.parse_shape(c, 'N C H W')
            Y, X = torch.meshgrid(torch.arange(start=0, end=crop_shape['H'], step=1), torch.arange(start=0, end=crop_shape['W'], step=1), indexing='ij')

            X_template = torch.min(torch.stack([X, torch.fliplr(X)], dim=0), dim=0)[0]
            Y_template = torch.min(torch.stack([Y, torch.flipud(Y)], dim=0), dim=0)[0]
            dist = X_template ** 2 + Y_template ** 2
            distance_maps.append(einops.rearrange(dist, 'H W -> 1 1 H W'))
        return distance_maps

    def blend_predictions(self, predictions_crops, scaled_image_shape, start_crops, sigma_crops=None, device=None):
        position_masks = self.create_position_masks(scaled_image_shape, start_crops, device=device)
        overlap_count = self.create_overlap_map(position_masks)
        placed_crops = self.place_crops_to_correct_position(scaled_image_shape, predictions_crops, start_crops, device=device)

        if self.scf_blending_method == BlendingMethod.AVERAGE:
            return self.blend_average(placed_crops, overlap_count)
        elif self.scf_blending_method == BlendingMethod.RANDOM:
            random_maps = self.generate_random_maps(position_masks, overlap_count)
            return self.blend_weight(placed_crops, random_maps)
        elif self.scf_blending_method == BlendingMethod.CONFIDENCE:
            placed_sigma_weight_maps = self.convert_sigmas_to_weights(scaled_image_shape, sigma_crops, start_crops, device=device)
            return self.blend_weight(placed_crops, placed_sigma_weight_maps)
        elif self.scf_blending_method == BlendingMethod.FURTHER_FROM_EDGE:
            distance_maps_crops = self.compute_distance_from_edge(predictions_crops)
            placed_dist_maps = self.place_crops_to_correct_position(scaled_image_shape, distance_maps_crops, start_crops, device=device)
            return self.blend_weight(placed_crops, placed_dist_maps)
        else:
            raise NotImplementedError(f'Blending mode {self.scf_blending_method} is not implemented')


    def downscale_predictions(self, predictions):
        down_predictions = {}
        for key, value in predictions.items():
            pass
        return down_predictions

    def concat_outputs(self, prediction_crops, keys):
        prediction_crops_concat = []
        for pc in prediction_crops:
            c_pred_crops = [pc[k] for k in keys]
            prediction_crops_concat.append(torch.cat(c_pred_crops, dim=1))
        return prediction_crops_concat

    def split_outputs(self, prediction_cat, dims, keys):
        # split outputs
        sum_layers = 0
        predictions = {}
        for k in keys:
            predictions[k] = prediction_cat[:, sum_layers:(sum_layers + dims[k]), :, :]
            sum_layers += dims[k]
        return predictions

    def correct_magnitudes(self, predictions, orig_image_shape, scaled_image_shape):
        predictions['flow'][:,0,:,:] = predictions['flow'][:,0,:,:] * orig_image_shape['W'] / scaled_image_shape['W']
        predictions['flow'][:,1,:,:] = predictions['flow'][:,1,:,:] * orig_image_shape['H'] / scaled_image_shape['H']
        if 'uncertainty' in predictions:
            predictions['uncertainty'] = predictions['uncertainty'] / self.scf_scale
        return predictions

    def forward(self, image1, image2, **kwargs):
        if not self.scf_enabled:
            return super(RAFTSCF, self).forward(image1, image2, **kwargs)

        orig_image_shape = einops.parse_shape(image1, 'N C H W')
        image1_scaled = torch.nn.functional.interpolate(image1, scale_factor=self.scf_scale, mode=self.scf_interpolate_type, align_corners=self.scf_align_corners)
        image2_scaled = torch.nn.functional.interpolate(image2, scale_factor=self.scf_scale, mode=self.scf_interpolate_type, align_corners=self.scf_align_corners)
        scaled_image_shape = einops.parse_shape(image1_scaled, 'N C H W')

        start_crops = self.calculate_start_crop_windows(image1_scaled)
        image1_crops = self.create_crops(image1_scaled, start_crops)
        image2_crops = self.create_crops(image2_scaled, start_crops)

        N_crops = len(image1_crops)
        prediction_crops = [super(RAFTSCF, self).forward(image1_crops[idx], image2_crops[idx], **kwargs) for idx in range(N_crops)]
        device = prediction_crops[0]['flow'].device

        if self.scf_debug:
            for idx, c_img1 in enumerate(image1_crops):
                pc = c_img1 * 1.0
                if idx >= 3:
                    pc[:, 2, :, :] = 0.
                pc[:, idx % 3, :, :] = 0.
                prediction_crops[idx]['test_image'] = pc


        # concatenating all ouputs for easier blending
        keys = list(prediction_crops[0].keys())
        keys = [k for k in keys if k != 'coords'] # remove coords from keys - they have different size, not currently implemented
        dims = {k:v.shape[1] for k,v in prediction_crops[0].items()}

        prediction_crops_concat = self.concat_outputs(prediction_crops, keys)
        sigma_crops = [torch.exp(pc['uncertainty']) for pc in prediction_crops]

        # BLENDING/MERGING/FUSION
        prediction_scaled = self.blend_predictions(prediction_crops_concat, scaled_image_shape, start_crops, sigma_crops=sigma_crops, device=device)
        # prediction_cat = torch.nn.functional.interpolate(prediction_scaled, size=(orig_image_shape['H'], orig_image_shape['W']), mode=self.scf_interpolate_type, scf_align_corners=self.scf_align_corners)
        prediction_cat = torch.nn.functional.interpolate(prediction_scaled, size=(orig_image_shape['H'], orig_image_shape['W']), mode='bilinear',
                                                         align_corners=self.scf_align_corners)

        predictions = self.split_outputs(prediction_cat, dims, keys)
        predictions = self.correct_magnitudes(predictions, orig_image_shape, scaled_image_shape)
        return predictions
