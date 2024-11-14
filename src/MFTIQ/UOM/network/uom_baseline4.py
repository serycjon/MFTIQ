# MFTIQ - WACV2025
import einops
import torch
import torch.nn as nn

from collections import OrderedDict

from MFTIQ.UOM.utils.image_manipulation import CenterPadding, universal_warp_backward, normalise_flow_and_coords
from MFTIQ.utils.misc import dummy_profile

from MFTIQ.UOM.heads.baseline_head import BaselineHead
from MFTIQ.UOM.heads.uncthreshold_head import UncertaintyHead, UncertaintyStackHead
from MFTIQ.UOM.building_bloks.basic_residual_block import BasicResidualBlock
from MFTIQ.UOM.features.pretrained_features_extractor_baseline4 import PretrainedFeaturesExtractorBaseline4
from MFTIQ.UOM.features.optical_flow_feature_extractor import OpticalFlowFeatureExtractorBaseline4
from MFTIQ.UOM.utils.image_manipulation import dot_product, compute_diff, DownsampleFlow
from MFTIQ.UOM.features.upsampler import UOMUpsampler
from spatial_correlation_sampler import SpatialCorrelationSampler

import os
if os.getenv('REMOTE_DEBUG'):
    import MFT.UOM.utils.plotting as dp # debugging plots


from rich.console import Console
console = Console()

profile = dummy_profile()


class UOMNetBase4(nn.Module):
    """
    Work on the 1/4 of image resolution DIRECTLY (baseline3 starts with full resolution and downsample with strided
    convolution, baseline4 downsamples with bilinear downsampling and then everything is done in 1/4 resolution)
    Use DINOv2 SMALL 14reg
    Bilinear upsampling of outputs (future: Upsampling to full resolution by the similar network as in RAFT)
    Optical flow is normalised before features are computed
    Dilatation convolutions to see bigger context in decoders
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.n_features = kwargs.get('uom_features', 32)
        self.use_group_norm = kwargs.get('uom_use_group_norm', True)
        self.downsample_coef = kwargs.get('downsample_coef', 1)
        self.head_type = kwargs.get('head_type', None)
        self.cost_volume_displacement = kwargs.get('cost_volume_displacement', None)
        self.single_logits = kwargs.get('single_logits', False)
        self.minimal_padding_coef = kwargs.get('padding_coef', 8)

        self.with_monodepth = kwargs.get('with_monodepth', False)
        self.with_resnet = kwargs.get('with_resnet', True)
        self.with_vggnet = kwargs.get('with_vggnet', False)
        self.with_dinov2 = kwargs.get('with_dinov2', True)
        self.with_imgcnn = kwargs.get('with_imgcnn', True)
        self.with_ofnet = kwargs.get('with_ofnet', True)

        self.upsample_type = kwargs.get('upsample_type', 'learned')
        self.with_spatial_correlation_sampler = kwargs.get('with_spatial_correlation_sampler', False)
        self.correlation_patch_size = kwargs.get('correlation_patch_size', 7)

        assert self.minimal_padding_coef in [8, 16, 32, 64]

        # TODO: should be read from feature extractor
        self.feature_names = []
        if self.with_resnet:
            self.feature_names = ['fine8', 'fine4', 'fine2']
        if self.with_vggnet:
            self.feature_names.extend(['vgg8', 'vgg4', 'vgg2'])
        if self.with_dinov2:
            self.feature_names.append('coarse')
        if self.with_imgcnn:
            self.feature_names.append('img')
        self.feature_names.append('aggreg')
        if self.with_monodepth:
            self.feature_names.append('monodepth')

        self.freeze_image_feature_extractors = kwargs.get('freeze_image_feature_extractors', False)

        # if init_nets==False -> do not init networks
        if not kwargs.get('init_nets', True):
            raise NotImplementedError

        self.padding_coef = max(self.minimal_padding_coef, self.downsample_coef)
        self.padding_features = CenterPadding(multiple=self.padding_coef)
        self.upsample_features = UOMUpsampler(upsample_type=self.upsample_type, upsample_coef=self.downsample_coef,
                                              orig_channels=5, final_channels=1, hidden_channels=self.n_features,
                                              align_corners=False)
        self.downsample_optical_flow = DownsampleFlow(downsample_factor=self.downsample_coef, align_corners=False)

        # FEATURE EXTRACTORS - input features extractors
        if self.with_ofnet:
            self.optical_flow_feature_extractor = OpticalFlowFeatureExtractorBaseline4(in_channels=4,
                                                                                   n_features=self.n_features,
                                                                                   downsample_coef=self.downsample_coef)

        if self.with_spatial_correlation_sampler:
            self.correlation_sampler = SpatialCorrelationSampler(kernel_size=1, patch_size=self.correlation_patch_size, stride=1, padding=0, dilation=1)
            n_corr_channels = self.correlation_patch_size ** 2
        else:
            self.correlation_sampler = None
            n_corr_channels = 1


        # NECK
        sim_features = n_corr_channels * len(self.feature_names) + len(self.feature_names)
        compressed_features = 2 * self.n_features
        optical_flow_features = 2 * self.n_features if self.with_ofnet else 0
        num_neck_features = sim_features + optical_flow_features + compressed_features
        self.neck_net = BasicResidualBlock(in_channels=num_neck_features, out_channels=self.n_features * 2,
                                           kernel_size=3, stride=1, padding=1, bias=False)
        # HEADS
        self.head_net = self.create_head_network(in_channels=self.n_features * 2,
                                                 n_features=self.n_features,
                                                 **kwargs)

        self.init_weights()

        # Not sure if it has to be last due to init_weights
        self.feature_extractors_compressors = PretrainedFeaturesExtractorBaseline4(
            n_features=self.n_features, downsample_coef=self.downsample_coef,
            minimal_padding_coef=self.minimal_padding_coef, with_monodepth=self.with_monodepth,
            with_imgcnn=self.with_imgcnn,
            with_resnet=self.with_resnet,
            with_dinov2=self.with_dinov2,
            with_vggnet=self.with_vggnet,
        )

        if self.freeze_image_feature_extractors:
            self.feature_extractors_compressors.eval()
            self.feature_extractors_compressors.requires_grad_(False)


    def create_head_network(self, in_channels, n_features=None, head_type=None, **kwargs):
        if n_features is None:
            n_features = self.n_features

        if head_type == 'baseline':
            return BaselineHead(in_channels=in_channels, n_features=n_features, **kwargs)
        elif head_type == 'uncthreshold':
            return UncertaintyHead(in_channels=in_channels, n_features=n_features, **kwargs)
        elif head_type == 'uncthreshold_stack':
            return UncertaintyStackHead(in_channels=in_channels, n_features=n_features, **kwargs)
        else:
            raise NotImplementedError


    def forward(self, img1, img2, flow, inference_mode=False, *args, **kwargs):
        return self.forward_single(img1, img2, flow, inference_mode=inference_mode, *args, **kwargs)


    def forward_single(self, img1, img2, flow, inference_mode=False, *args, **kwargs):
        if inference_mode:
            return self.forward_inference(img1, img2, flow, inference_mode=inference_mode, *args, **kwargs)
        else:
            return self.forward_train(img1, img2, flow, inference_mode=inference_mode, *args, **kwargs)


    def forward_train(self, img1, img2, flow, inference_mode=False, *args, **kwargs):

        left_cache = kwargs.get('left_cache', {'image': img1})
        right_cache = kwargs.get('right_cache', {'image': img2})
        rotation = kwargs.get('rotation', 0)

        # extract and compress features -> values in cache are not computed again
        left_cache = self.feature_extractors_compressors(img1, left_cache)
        right_cache = self.feature_extractors_compressors(img2, right_cache, rotation)

        with torch.no_grad():
            flow_pad = self.padding_features(flow)
            img_orig_pad = self.padding_features(img1)
            flow_correct_size = self.downsample_optical_flow(flow_pad)
            norm_flow, coords1, coords2 = normalise_flow_and_coords(flow_pad, keep_shape=True)
            norm_flow_input = torch.concat([norm_flow, coords2], dim=1)

        if self.with_ofnet:
            f_of, g_of = self.optical_flow_feature_extractor(norm_flow_input)
        else:
            f_of, g_of = None, None

        # warp features
        left_upw_f = OrderedDict()
        right_upw_f = OrderedDict()
        cost_volume = OrderedDict()

        for k in self.feature_names:
            left_upw_f[k], right_upw_f[k], cost_volume[k] = universal_warp_backward(
                            left_cache[f'{k}_f'], right_cache[f'{k}_f'], flow_correct_size,
                            cost_volume_displacement=self.cost_volume_displacement
            )

        # dot products and subtraction of reference and warped target features
        similarities = self.compute_all_similarities(left_upw_f, right_upw_f, cost_volume)


        # concat all input features
        neck_input_features_list = []
        if self.with_ofnet:
            neck_input_features_list.extend([f_of, g_of])
        neck_input_features_list.extend([similarities, left_upw_f['aggreg'], right_upw_f['aggreg']])
        neck_input_features = torch.concat(neck_input_features_list, dim=1)

        # compute neck (common network for all heads)
        neck_features = self.neck_net(neck_input_features)

        # compute outputs (heads)
        head_outputs = self.head_net(neck_features, **kwargs)

        # prepare for outputs
        if self.downsample_coef != 1:
            features_for_filter = self.upsample_features.compute_filter_features(flow=norm_flow, image=img_orig_pad)
            outputs = {k: self.upsample_features(v, features_for_filter) for k, v in head_outputs.items()}
            outputs = {k: self.padding_features.unpad(v) for k, v in outputs.items()}
        else:
            outputs = head_outputs

        outputs['flow'] = flow

        # change outputs for inference
        if inference_mode:
            outputs = self.head_net.inference_outputs_formatter(outputs, **kwargs)

        outputs['left_cache'] = left_cache
        outputs['right_cache'] = right_cache
        return outputs


    def compute_all_similarities(self, left_data, right_data, cost_volume):
        similarities_list = []
        for k, v in left_data.items():
            if self.with_spatial_correlation_sampler:
                b, c, h, w = v.shape
                c_sim = self.correlation_sampler(v, right_data[k]).view(b, -1, h, w)
                similarities_list.append(c_sim)
            else:
                similarities_list.append(dot_product(v, right_data[k]))
            similarities_list.append(compute_diff(v, right_data[k], abs=True))
            if cost_volume is not None and cost_volume[k] is not None:
                similarities_list.append(cost_volume[k])
        return torch.cat(similarities_list, dim=1)


    def init_memory(self):
        self.outputs_memory = []

    def detach_memory(self):
        self.tc_net.detach_memory()

    @torch.inference_mode()
    @profile
    def forward_inference(self, img1, img2, flow, inference_mode=True, *args, **kwargs):
        assert inference_mode

        additional_inputs = {}

        H, W = img1.shape[2], img1.shape[3]
        outputs = self.forward_train(img1, img2, flow, inference_mode=inference_mode, *args, **kwargs, **additional_inputs)

        assert outputs['occlusion'].shape == (1, H, W)
        assert outputs['uncertainty'].shape == (1, H, W)
        return outputs

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
