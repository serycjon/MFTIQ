# MFTIQ - WACV2025
import torch
import torch.nn as nn
import einops
import torch
import torchvision.transforms as T

from MFTIQ.utils.misc import dummy_profile
from MFTIQ.UOM.utils.image_manipulation import CenterPadding
from MFTIQ.UOM.building_bloks.basic_residual_block import BasicResidualBlock
from MFTIQ.UOM.features.feature_compressor import FeatureCompressor
import os
if os.getenv('REMOTE_DEBUG'):
    import MFT.UOM.utils.plotting as dp

profile = dummy_profile()


class PretrainedFeaturesExtractorBaseline4(nn.Module):
    def __init__(self, n_features, downsample_coef=4, minimal_padding_coef=8,
                 with_monodepth=False, with_resnet=True, with_dinov2=True, with_imgcnn=True, with_vggnet=False,
                 with_hiera=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_features = n_features
        self.downsample_coef = downsample_coef
        self.minimal_padding_coef = minimal_padding_coef

        self.with_monodepth = with_monodepth
        self.with_resnet = with_resnet
        self.with_dinov2 = with_dinov2
        self.with_imgcnn = with_imgcnn
        self.with_vggnet = with_vggnet
        self.with_hiera = with_hiera

        n_compressors = 0
        if self.with_resnet:
            self.fine_feature_compressor0 = FeatureCompressor(in_channels=64, out_channels=self.n_features, in_downsample=4, out_downsample=self.downsample_coef, minimal_padding_coef=self.minimal_padding_coef)
            self.fine_feature_compressor1 = FeatureCompressor(in_channels=256, out_channels=self.n_features, in_downsample=4, out_downsample=self.downsample_coef, minimal_padding_coef=self.minimal_padding_coef)
            self.fine_feature_compressor2 = FeatureCompressor(in_channels=512, out_channels=self.n_features, in_downsample=8, out_downsample=self.downsample_coef, minimal_padding_coef=self.minimal_padding_coef)
            n_compressors += 3
        if self.with_vggnet:
            self.vgg_feature_compressor0 = FeatureCompressor(in_channels=128, out_channels=self.n_features,
                                                              in_downsample=2, out_downsample=self.downsample_coef,
                                                              minimal_padding_coef=self.minimal_padding_coef)
            self.vgg_feature_compressor1 = FeatureCompressor(in_channels=256, out_channels=self.n_features,
                                                              in_downsample=4, out_downsample=self.downsample_coef,
                                                              minimal_padding_coef=self.minimal_padding_coef)
            self.vgg_feature_compressor2 = FeatureCompressor(in_channels=512, out_channels=self.n_features,
                                                              in_downsample=8, out_downsample=self.downsample_coef,
                                                              minimal_padding_coef=self.minimal_padding_coef)
            n_compressors += 3
        if self.with_dinov2:
            self.coarse_feature_compressor = FeatureCompressor(in_channels=384, out_channels=self.n_features, in_downsample=14, out_downsample=self.downsample_coef, minimal_padding_coef=self.minimal_padding_coef)
            n_compressors += 1
        if self.with_imgcnn:
            self.image_feature_compressor = FeatureCompressor(in_channels=3, out_channels=self.n_features, in_downsample=1, out_downsample=self.downsample_coef, minimal_padding_coef=self.minimal_padding_coef)
            n_compressors += 1

        if self.with_monodepth:
            self.monodepth_feature_compressor = FeatureCompressor(in_channels=6, out_channels=self.n_features, in_downsample=1, out_downsample=self.downsample_coef, minimal_padding_coef=self.minimal_padding_coef)
            n_compressors += 1

        if self.with_hiera:
            self.hiera_feature_compressor = FeatureCompressor(in_channels=6, out_channels=self.n_features,
                                                                  in_downsample=1, out_downsample=self.downsample_coef,
                                                                  minimal_padding_coef=self.minimal_padding_coef)
            n_compressors += 1

        self.aggregation_compressor = nn.Sequential(
            BasicResidualBlock(n_compressors * self.n_features, 2 * self.n_features, kernel_size=3, stride=1, padding=1),
            BasicResidualBlock(2 * self.n_features, 1 * self.n_features, kernel_size=3, stride=1, padding=1),
        )

        assert n_compressors > 0, """There should be at least 1 feature extractor/compression network."""
        self.init_weights()


        if self.with_resnet:
            # FINE FEATURES FROM FIRST TWO LAYERS OF RESNET-50
            fine_feature_backbone_whole = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")
            fine_feature_backbone_whole.eval()
            fine_feature_backbone_whole.requires_grad_(False)

            if downsample_coef in [4, 8]:
                self.fine_feature_backbone_layer0 = nn.Sequential(*list(fine_feature_backbone_whole.children())[:4])  # four input layers
                self.fine_feature_backbone_layer1 = nn.Sequential(*list(fine_feature_backbone_whole.children())[4])  # first layer (from four)
                self.fine_feature_backbone_layer2 = nn.Sequential(*list(fine_feature_backbone_whole.children())[5])  # second layer (from four)
            else:
                raise NotImplementedError

            self.fine_patch_size = max(8, self.downsample_coef)
            self.padding_fine = CenterPadding(multiple=self.fine_patch_size)
            self.fine_transform = T.Compose([T.Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375))])

        if self.with_vggnet:
            vgg_feature_backbone_whole = torch.hub.load("pytorch/vision", "vgg19", pretrained=True)
            vgg_feature_backbone_whole.eval()
            vgg_feature_backbone_whole.requires_grad_(False)

            if downsample_coef in [4, 8]:
                self.vgg_feature_backbone_layer0 = nn.Sequential(*list(vgg_feature_backbone_whole.children())[0][:8])  # four input layers
                self.vgg_feature_backbone_layer1 = nn.Sequential(*list(vgg_feature_backbone_whole.children())[0][8:17])
                self.vgg_feature_backbone_layer2 = nn.Sequential(*list(vgg_feature_backbone_whole.children())[0][17:26])# first layer (from four)
            else:
                raise NotImplementedError

            self.vgg_patch_size = max(8, self.downsample_coef)
            self.padding_vgg = CenterPadding(multiple=self.vgg_patch_size)
            self.vgg_transform = T.Compose([T.Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375))])


        if self.with_imgcnn:
            self.img_patch_size = max(8, self.downsample_coef)
            self.padding_img = CenterPadding(multiple=self.img_patch_size)
            self.img_transform = T.Compose([T.Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375))])

        if self.with_dinov2:
            # COARSE FEATURES FROM FIRST TWO LAYERS OF DINOv2
            self.coarse_feature_backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
            self.coarse_feature_backbone.eval()
            self.coarse_feature_backbone.requires_grad_(False)
            self.coarse_patch_size = self.coarse_feature_backbone.patch_size
            self.padding_coarse = CenterPadding(multiple=self.coarse_patch_size)
            self.coarse_transform = T.Compose([T.Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375))])
            # self.coarse_upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2')
            # self.coarse_upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=True)

        # MONODEPTH NETWORK (and NORMALS)
        if self.with_monodepth:
            self.monodepth_network = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
            self.monodepth_network.eval()
            self.monodepth_network.requires_grad_(False)
            self.monodepth_patch_size = self.monodepth_network.depth_model.encoder.patch_size # not sure
            self.monodepth_compress_padding_coef = max(8, self.downsample_coef)
            self.padding_monodepth = CenterPadding(multiple=self.monodepth_patch_size)
            self.padding_monodepth_compress = CenterPadding(multiple=self.monodepth_compress_padding_coef)
            self.monodepth_transform = T.Compose([T.Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375))])

        # HIERA (from SAMv2)
        if self.with_hiera:
            hiera_feature_backbone_whole = torch.hub.load("facebookresearch/hiera",
                                                          model="mae_hiera_base_224", pretrained=True,
                                                          checkpoint="mae_in1k")
            hiera_feature_backbone_whole.eval()
            hiera_feature_backbone_whole.requires_grad_(False)
            self.hiera_patch_size = self.hiera_feature_backbone.patch_size
            self.padding_hiera = CenterPadding(multiple=self.hiera_patch_size)
            self.hiera_transform = T.Compose(
                [T.Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375))])


    def forward(self, img, cache=None, rotation=0):
        cache = cache if cache is not None else {}
        assert isinstance(cache, dict)

        # rotate input image
        if rotation == 0:
            pass
        elif rotation == 180:
            img = torch.rot90(img, 2, dims=[2,3])
        elif rotation == 90:
            img = torch.rot90(img, 1, dims=[2,3])
        elif rotation == 270:
            img = torch.rot90(img, 3, dims=[2,3])
        else:
            raise NotImplementedError(
                f'Rotation implemented only for 0, 90, 180 and 270 degrees, degrees={self.degrees} not implemented')
        rotate_features_names = []

        if self.with_resnet:
            self.padding_fine.init_padding(img)

        if self.with_dinov2:
            self.padding_coarse.init_padding(img)

        if self.with_imgcnn:
            self.padding_img.init_padding(img)

        if self.with_monodepth:
            self.padding_monodepth.init_padding(img)

        if self.with_vggnet:
            self.padding_vgg.init_padding(img)

        if self.with_resnet and ('fine2_f' not in cache or 'fine4_f' not in cache or 'fine8_f' not in cache):
            cache['fine2_f'], cache['fine4_f'], cache['fine8_f'] = self.compute_fine_features(img)
            rotate_features_names.extend(['fine2_f', 'fine4_f', 'fine8_f'])
        if self.with_vggnet and ('vgg4_f' not in cache or 'vgg8_f' not in cache):
            cache['vgg2_f'], cache['vgg4_f'], cache['vgg8_f'] = self.compute_vgg_features(img)
            rotate_features_names.extend(['vgg2_f', 'vgg4_f', 'vgg8_f'])
        if self.with_dinov2 and 'coarse_f' not in cache:
            cache['coarse_f'] = self.compute_coarse_features(img)
            rotate_features_names.extend(['coarse_f'])
        if self.with_imgcnn and 'img_f' not in cache:
            cache['img_f'] = self.compute_img_features(img)
            rotate_features_names.extend(['img_f'])
        if self.with_monodepth and 'monodepth_f' not in cache:
            cache['monodepth_f'], monodepth_dict = self.compute_monodepth_features(img)
            cache['depth'] = monodepth_dict['monodepth']
            cache['depth_conf'] = monodepth_dict['monodepth_confidence']
            cache['normals'] = monodepth_dict['normals']
            cache['normals_conf'] = monodepth_dict['normals_confidence']
            rotate_features_names.extend(['monodepth_f', 'depth', 'depth_conf', 'normals', 'normals_conf'])
        if self.with_hiera and 'hiera_f' not in cache:
            cache['hiera_f'] = self.compute_hiera_features(img)
            rotate_features_names.extend(['hiera_f'])

        if 'aggreg_f' not in cache:
            agg_inpute_list = [cache[k] for k in ['fine2_f','fine4_f','fine8_f','vgg2_f','vgg4_f','vgg8_f','coarse_f','img_f','monodepth_f', 'hiera_f'] if k in cache]
            agg_input = torch.cat(agg_inpute_list, dim=1)
            cache['aggreg_f'] = self.aggregation_compressor(agg_input)
            rotate_features_names.extend(['aggreg_f'])

        if rotation == 0:
            return cache

        # rotate back
        for key in rotate_features_names:
            if '_f' not in key:
                continue

            value = cache[key]
            if rotation == 180:
                value = torch.rot90(value, 2, dims=[2, 3])
            elif rotation == 90:
                value = torch.rot90(value, 3, dims=[2, 3])
            elif rotation == 270:
                value = torch.rot90(value, 1, dims=[2, 3])
            else:
                raise NotImplementedError(
                    f'Rotation implemented only for 0, 90, 180 and 270 degrees, degrees={self.degrees} not implemented')
            cache[key] = value

        return cache


    def compute_aggregation(self, data):
        return self.aggregation_compressor(data)


    def get_pad(self, position, feature_name, feature=None):
        assert position in ['left', 'top']

        if 'coarse' in feature_name:
            if feature is not None:
                self.padding_coarse.init_padding(feature)

            if position == 'top':
                return self.padding_coarse.get_top_pad()
            else:
                return self.padding_coarse.get_left_pad()

        elif 'fine' in feature_name:
            if feature is not None:
                self.padding_fine.init_padding(feature)

            if position == 'top':
                return self.padding_fine.get_top_pad()
            else:
                return self.padding_fine.get_left_pad()
        else:
            raise NotImplementedError


    @profile
    def compute_fine_features(self, img, **kwargs):
        with torch.no_grad():
            self.fine_feature_backbone_layer1.eval()
            self.fine_feature_backbone_layer2.eval()

            img_tr = self.fine_transform(img)
            img_pad = self.padding_fine(img_tr)
            features0 = self.fine_feature_backbone_layer0(img_pad)
            features1 = self.fine_feature_backbone_layer1(features0)
            features2 = self.fine_feature_backbone_layer2(features1)

        feat2_compress = self.fine_feature_compressor0(features0)
        feat4_compress = self.fine_feature_compressor1(features1)
        feat8_compress = self.fine_feature_compressor2(features2)
        return feat2_compress, feat4_compress, feat8_compress


    def compute_vgg_features(self, img, **kwargs):
        with torch.no_grad():
            self.vgg_feature_backbone_layer0.eval()
            self.vgg_feature_backbone_layer1.eval()
            self.vgg_feature_backbone_layer2.eval()

            img_tr = self.vgg_transform(img)
            img_pad = self.padding_vgg(img_tr)
            features0 = self.vgg_feature_backbone_layer0(img_pad)
            features1 = self.vgg_feature_backbone_layer1(features0)
            features2 = self.vgg_feature_backbone_layer2(features1)

        vgg2_compress = self.vgg_feature_compressor0(features0)
        vgg4_compress = self.vgg_feature_compressor1(features1)
        vgg8_compress = self.vgg_feature_compressor2(features2)
        return vgg2_compress, vgg4_compress, vgg8_compress


    def compute_img_features(self, img, **kwargs):
        with torch.no_grad():
            img_tr = self.img_transform(img)
            img_pad = self.padding_img(img_tr)
        return self.image_feature_compressor(img_pad)


    def compute_monodepth_features(self, img, **kwargs):
        with torch.no_grad():
            self.monodepth_network.eval()
            img_tr = self.monodepth_transform(img)
            img_pad = self.padding_monodepth(img_tr)
            pred_depth, confidence, output_dict = self.monodepth_network.inference({'input': img_pad})
            pred_normal = output_dict['prediction_normal'][:, :3, :, :]
            normal_confidence = output_dict['prediction_normal'][:, 3:4, :, :]
            pred_depth, confidence, pred_normal, normal_confidence = self.padding_monodepth.unpad(
                pred_depth, confidence, pred_normal, normal_confidence)
            input_features = torch.cat([pred_depth, confidence, pred_normal, normal_confidence], dim=1)
            input_features_pad = self.padding_monodepth_compress(input_features)

        monodepth_compress = self.monodepth_feature_compressor(input_features_pad)
        return monodepth_compress, {
            'monodepth': pred_depth, 'monodepth_confidence': confidence,
            'normals': pred_normal, 'normals_confidence': normal_confidence
        }


    @profile
    def compute_coarse_features(self, img, **kwargs):
        with torch.no_grad():
            self.coarse_feature_backbone.eval()

            img_tr = self.coarse_transform(img)
            img_pad = self.padding_coarse(img_tr)
            img_pad_shape = einops.parse_shape(img_pad, 'b c h w')
            features_flat = self.coarse_feature_backbone.get_intermediate_layers(img_pad)[0]
            features = einops.rearrange(features_flat, 'b (h w) c -> b c h w', h=int(img_pad_shape['h']/self.coarse_patch_size), w=int(img_pad_shape['w']/self.coarse_patch_size))

            # features_ups = self.coarse_upsampler(img_tr)
            # img_non_norm_pad = self.padding_coarse(img)
            # features_ups = self.coarse_upsampler(img_pad)

        features_out = self.coarse_feature_compressor(features, orig_img=img)
        return features_out


    @profile
    def compute_hiera_features(self, img, **kwargs):
        with torch.no_grad():
            self.hiera_feature_backbone.eval()

            img_tr = self.hiera_transform(img)
            img_pad = self.padding_hiera(img_tr)
            img_pad_shape = einops.parse_shape(img_pad, 'b c h w')
            features_flat = self.hiera_feature_backbone.get_intermediate_layers(img_pad)[0]
            features = einops.rearrange(features_flat, 'b (h w) c -> b c h w', h=int(img_pad_shape['h']/self.hiera_patch_size), w=int(img_pad_shape['w']/self.hiera_patch_size))

        features_out = self.hiera_feature_compressor(features, orig_img=img)
        return features_out



    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


