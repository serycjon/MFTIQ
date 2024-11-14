# MFTIQ - WACV2025
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, bias=True, relu=True,
                    dilation=1, use_group_norm=None, stride=1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_group_norm = True
        self.layer = self.basic_block(in_channels, out_channels, kernel_size=kernel_size,
                                      padding=padding, bias=bias, relu=relu, dilation=dilation,
                                      use_group_norm=use_group_norm, stride=stride, **kwargs)

    def forward(self, x):
        return self.layer(x)


    def basic_block(self, in_channels, out_channels, kernel_size=1, padding=0, bias=True, relu=True,
                    dilation=1, use_group_norm=None, stride=1, **kwargs):
        num_groups = kwargs.get('num_groups', max(1, out_channels // 8))
        if use_group_norm is None:
            use_group_norm = self.use_group_norm
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                            bias=bias, stride=stride, dilation=dilation)]

        if use_group_norm:
            layers.append(nn.GroupNorm(num_groups=num_groups, num_channels=out_channels))
        if relu:
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)