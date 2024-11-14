# MFTIQ - WACV2025
import torch
import torch.nn as nn

class BasicResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, bias=False,
                    dilation=1, use_group_norm=True, stride=1, padding_mode='zeros', *args, **kwargs):
        self.num_groups = kwargs.pop('num_groups', out_channels // 8)

        super().__init__(*args, **kwargs)

        self.use_group_norm = use_group_norm

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               padding_mode=padding_mode, bias=bias, dilation=dilation)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias,
                               padding_mode=padding_mode, dilation=dilation)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

        if self.use_group_norm:
            self.norm = nn.GroupNorm(num_groups=self.num_groups, num_channels=out_channels)
        else:
            raise NotImplementedError

    def forward(self, x):
        # from NEUFLOW
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        return self.norm(x1 + x2)