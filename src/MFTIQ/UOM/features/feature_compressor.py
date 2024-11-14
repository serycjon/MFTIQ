# MFTIQ - WACV2025
import torch
import torch.nn as nn
import torch.nn.functional as F
from MFTIQ.UOM.building_bloks.basic_residual_block import BasicResidualBlock
from MFTIQ.UOM.utils.image_manipulation import CenterPadding



class FeatureCompressor(nn.Module):
    def __init__(self, in_channels, out_channels, in_downsample, out_downsample, minimal_padding_coef=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_downsample = in_downsample
        self.out_downsample = out_downsample
        self.minimal_padding_coef = minimal_padding_coef

        if self.out_downsample not in [1,2,4,8]:
            raise NotImplementedError

        current_downsample = self.in_downsample
        self.forward_sequence = nn.ModuleList()
        self.final_layer = None
        self.upsample_special = None

        self.first_compressor_block = BasicResidualBlock(in_channels, out_channels, kernel_size=1, padding=0, stride=1)
        if self.in_downsample == self.out_downsample:
            self.final_layer = BasicResidualBlock(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
            return

        if self.in_downsample not in [1, 2, 4, 8, 16, 32]:
            current_downsample = 1
            self.upsample_special = nn.Upsample(scale_factor=self.in_downsample, mode='bilinear', align_corners=False)
            self.padding_orig = CenterPadding(multiple=self.in_downsample)
            self.padding_new = CenterPadding(multiple=max(self.out_downsample, self.minimal_padding_coef))
        elif current_downsample > self.out_downsample:
            self.upsample_special = nn.Upsample(scale_factor=int(round(self.in_downsample/self.out_downsample)), mode='bilinear', align_corners=False)
            self.final_layer = BasicResidualBlock(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
            return

        channels_multiplication = 0
        if current_downsample <= 1 <= self.out_downsample:
            channels_multiplication += 1
            # for downsample 1,2,4,8
            self.forward_sequence.append(
                BasicResidualBlock(self.out_channels, self.out_channels, kernel_size=self.out_downsample, stride=self.out_downsample, padding=0))

        if current_downsample <= 2 <= self.out_downsample:
            channels_multiplication += 1
            if self.out_downsample == 8:
                self.forward_sequence.append(BasicResidualBlock(self.out_channels, self.out_channels, kernel_size=8, stride=4, padding=2))
            elif self.out_downsample == 4:
                self.forward_sequence.append(BasicResidualBlock(self.out_channels, self.out_channels, kernel_size=8, stride=2, padding=3))
            elif self.out_downsample == 2:
                self.forward_sequence.append(BasicResidualBlock(self.out_channels, self.out_channels, kernel_size=7, stride=1, padding=3))

        if current_downsample <= 4 <= self.out_downsample:
            channels_multiplication += 1
            if self.out_downsample == 8:
                self.forward_sequence.append(BasicResidualBlock(self.out_channels, self.out_channels, kernel_size=8, stride=2, padding=3))
            elif self.out_downsample == 4:
                self.forward_sequence.append(BasicResidualBlock(self.out_channels, self.out_channels, kernel_size=7, stride=1, padding=3))

        if current_downsample <= 8 <= self.out_downsample:
            channels_multiplication += 1
            self.forward_sequence.append(
                BasicResidualBlock(self.out_channels, self.out_channels, kernel_size=7, stride=1, padding=3))

        self.final_layer = BasicResidualBlock(out_channels * channels_multiplication, out_channels, kernel_size=3, padding=1, stride=1)



    def forward(self, x, orig_img=None):

        x = self.first_compressor_block(x)

        if self.upsample_special is not None:
            x = self.upsample_special(x)

        if self.in_downsample not in [1, 2, 4, 8, 16]:
            assert orig_img is not None
            self.padding_orig.init_padding(orig_img)
            x = self.padding_orig.unpad(x)
            x = self.padding_new(x)

        if len(self.forward_sequence) > 0:
            feature_list = []
            for c_layer in self.forward_sequence:
                feature_list.append(c_layer(x))
                x = F.avg_pool2d(x, kernel_size=2, stride=2)
            x = torch.cat(feature_list, dim=1)

        x = self.final_layer(x)
        return x
