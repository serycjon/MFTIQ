# MFTIQ - WACV2025
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from MFTIQ.UOM.building_bloks.basic_residual_block import BasicResidualBlock

class UOMUpsampler(nn.Module):

    def __init__(self,
                 upsample_type='bilinear',
                 upsample_coef=1,
                 align_corners=False,
                 orig_channels=None,
                 hidden_channels=None,
                 final_channels=1,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.upsample_type = upsample_type
        self.upsample_coef = upsample_coef
        self.align_corners = align_corners

        self.orig_channels = orig_channels
        self.hidden_channels = hidden_channels
        self.final_channels = final_channels

        self.main_upsampler = None
        self.feature_block = None
        self.img_transform = T.Compose([T.Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375))])

        if self.upsample_coef == 1:
            pass

        if self.upsample_type == 'bilinear':
            self.main_upsampler = nn.Upsample(scale_factor=upsample_coef, mode='bilinear', align_corners=align_corners)
        elif self.upsample_type == 'learned':
            assert orig_channels is not None
            assert hidden_channels is not None
            assert final_channels is not None

            self.feature_block = BasicResidualBlock(self.orig_channels, self.hidden_channels,
                                                    kernel_size=self.upsample_coef, stride=self.upsample_coef,
                                                    padding=0)
            self.main_upsampler = UpSample(self.hidden_channels, self.upsample_coef, self.final_channels)
        else:
            raise NotImplementedError


    def compute_filter_features(self, image, flow):
        if self.upsample_type != 'learned':
            return None

        image_norm = self.img_transform(image)
        inputs = torch.cat([image_norm, flow], dim=1)

        return self.feature_block(inputs)

    def forward(self, x, source_filter_features=None):

        if self.upsample_coef == 1:
            return x
        elif self.upsample_type == 'bilinear':
            return self.main_upsampler(x)
        elif self.upsample_type == 'learned':
            assert source_filter_features is not None
            return self.main_upsampler(source_filter_features, x)
        else:
            raise NotImplementedError


class UpSample(torch.nn.Module):
    def __init__(self, feature_dim, upsample_factor, final_channels):
        super(UpSample, self).__init__()

        self.upsample_factor = upsample_factor
        self.final_channels = final_channels

        self.conv1 = torch.nn.Conv2d(self.final_channels + feature_dim, 256, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(256, 512, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(512, upsample_factor ** 2 * 9, 1, 1, 0)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, feature_source_filter, features_to_upsample):

        concat = torch.cat((features_to_upsample, feature_source_filter), dim=1)

        mask = self.conv3(self.relu(self.conv2(self.relu(self.conv1(concat)))))

        b, _, h, w = features_to_upsample.shape

        mask = mask.view(b, 1, 9, self.upsample_factor, self.upsample_factor, h, w)  # [B, 1, 9, K, K, H, W]
        mask = torch.softmax(mask, dim=2)

        upsampled_features = F.unfold(features_to_upsample, [3, 3], padding=1)
        upsampled_features = upsampled_features.view(b, self.final_channels, 9, 1, 1, h, w)  # [B, 2, 9, 1, 1, H, W]

        upsampled_features = torch.sum(mask * upsampled_features, dim=2)  # [B, 2, K, K, H, W]
        upsampled_features = upsampled_features.permute(0, 1, 4, 2, 5, 3)  # [B, 2, K, H, K, W]
        upsampled_features = upsampled_features.reshape(b, self.final_channels, self.upsample_factor * h,
                                  self.upsample_factor * w)  # [B, 2, K*H, K*W]

        return upsampled_features