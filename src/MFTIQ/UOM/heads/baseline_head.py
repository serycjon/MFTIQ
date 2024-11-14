# MFTIQ - WACV2025
import torch
import torch.nn as nn
from MFTIQ.UOM.building_bloks.basic_block import BasicBlock

class HeadAbstract(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def define_base_head_net(self, in_channels, n_features=1, out_channels=2):
        return nn.Sequential(
            BasicBlock(in_channels, n_features, padding=1, kernel_size=3),
            BasicBlock(n_features, n_features, padding=1, kernel_size=3),
            BasicBlock(n_features, out_channels, padding=1, kernel_size=3, relu=False, use_group_norm=False),
        )

    def define_uncertainty_head_net(self, in_channels, n_features, out_channels=1):
        return self.define_base_head_net(in_channels, n_features, out_channels)

    def define_occlusion_head_net(self, in_channels, n_features=1, out_channels=2):
        return self.define_base_head_net(in_channels, n_features, out_channels)


class BaselineHead(HeadAbstract):
    def __init__(self, in_channels, n_features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_features = n_features
        self.in_channels = in_channels

        self.occlusion_head = self.define_occlusion_head_net(in_channels=self.in_channels, n_features=self.n_features)
        self.uncertainty_head = self.define_uncertainty_head_net(in_channels=self.in_channels, n_features=self.n_features)

    def forward(self, x, inference_mode=False, *args, **kwargs):
        occlusion_est = self.occlusion_head(x)
        uncertainty_est = self.uncertainty_head(x)
        outputs = {'occlusion': occlusion_est, 'uncertainty': uncertainty_est}

        if inference_mode:
            outputs['occlusion'] = outputs['occlusion'].softmax(dim=1)[0, 1:, :, :]
            sigma = torch.sqrt(torch.exp(outputs['uncertainty'][0, :, :, :]))
            outputs['uncertainty'] = sigma

        return outputs

