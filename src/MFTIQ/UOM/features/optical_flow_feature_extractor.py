# MFTIQ - WACV2025
import einops
import torch
import torch.nn as nn
import torchvision.transforms as T

from MFTIQ.UOM.building_bloks.basic_block import BasicBlock
from MFTIQ.UOM.building_bloks.basic_residual_block import BasicResidualBlock
from MFTIQ.UOM.features.feature_compressor import FeatureCompressor


class OpticalFlowFeatureExtractor(nn.Module):
    def __init__(self, in_channels, n_features, downsample_coef, *args, **kwargs):
        super().__init__()

        self.n_features = n_features
        self.use_group_norm = kwargs.get('use_group_norm', True)

        assert downsample_coef in [1, 2, 4, 8]

        self.net_list = [BasicBlock(in_channels, self.n_features, padding=0, kernel_size=1, stride=1,
                                    use_group_norm=False)]
        if downsample_coef >= 2:
            self.net_list.append(BasicBlock(self.n_features, 2 * self.n_features, padding=1, kernel_size=3, stride=2,
                                            use_group_norm=self.use_group_norm))
        else:
            self.net_list.append(BasicBlock(self.n_features, self.n_features, padding=1, kernel_size=3, stride=1,
                                            use_group_norm=self.use_group_norm))

        if downsample_coef >= 4:
            self.net_list.append(
                BasicBlock(2 * self.n_features, 4 * self.n_features, padding=1, kernel_size=3, stride=2,
                           use_group_norm=self.use_group_norm))
        else:
            self.net_list.append(
                BasicBlock(self.n_features, self.n_features, padding=1, kernel_size=3, stride=1,
                           use_group_norm=self.use_group_norm))

        if downsample_coef >= 8:
            self.net_list.append(
                BasicBlock(4 * self.n_features, 8 * self.n_features, padding=1, kernel_size=3, stride=2,
                           use_group_norm=self.use_group_norm))

        self.net_list = nn.ModuleList(self.net_list)

        self.resize = T.Resize((32, 32), antialias=True)
        self.global_conv_net = nn.Sequential(
            BasicBlock(downsample_coef * self.n_features, downsample_coef * self.n_features, padding=1, kernel_size=3, stride=2, use_group_norm=self.use_group_norm),
            BasicBlock(downsample_coef * self.n_features, downsample_coef * self.n_features, padding=1, kernel_size=3, stride=2, use_group_norm=self.use_group_norm),
        )
        self.global_linear_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(downsample_coef * self.n_features * 8 * 8, downsample_coef * self.n_features * self.n_features),
            nn.ReLU(),
            nn.Linear(downsample_coef * self.n_features * self.n_features, downsample_coef * self.n_features),
        )


    def forward(self, x):
        outputs = [x]
        for layer in self.net_list:
            outputs.append(layer(outputs[-1]))

        global_data_input = self.resize(outputs[-1])
        global_conv_data = self.global_conv_net(global_data_input)
        global_data = self.global_linear_net(global_conv_data)
        H, W = outputs[-1].shape[2:]
        global_output = einops.repeat(global_data, 'B C -> B C H W', H=H, W=W)

        return outputs[-1], global_output



class OpticalFlowFeatureExtractorBaseline4(nn.Module):
    def __init__(self, in_channels, n_features, downsample_coef, *args, **kwargs):
        super().__init__()

        self.n_features = n_features
        self.use_group_norm = kwargs.get('use_group_norm', True)

        assert downsample_coef in [1, 2, 4, 8]

        self.feature_extractor = FeatureCompressor(in_channels, out_channels=n_features, in_downsample=1,
                                                   out_downsample=downsample_coef)

        self.resize = T.Resize((32, 32), antialias=True)
        self.global_conv_net = nn.Sequential(
            BasicResidualBlock(self.n_features, self.n_features, padding=1, kernel_size=3, stride=2,
                       use_group_norm=self.use_group_norm),
            BasicResidualBlock(self.n_features, self.n_features, padding=1, kernel_size=3, stride=2,
                       use_group_norm=self.use_group_norm),
        )
        self.global_linear_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n_features * 8 * 8, self.n_features * self.n_features),
            nn.ReLU(),
            nn.Linear(self.n_features * self.n_features, self.n_features),
        )

    def forward(self, x):
        x = self.feature_extractor(x)

        global_data_input = self.resize(x)
        global_conv_data = self.global_conv_net(global_data_input)
        global_data = self.global_linear_net(global_conv_data)
        H, W = x.shape[2:]
        global_output = einops.repeat(global_data, 'B C -> B C H W', H=H, W=W)

        return x, global_output