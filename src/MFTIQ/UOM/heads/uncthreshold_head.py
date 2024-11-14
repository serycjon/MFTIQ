# MFTIQ - WACV2025
import torch
import torch.nn as nn
import torch.nn.functional as F
from MFTIQ.UOM.building_bloks.basic_block import BasicBlock
from MFTIQ.UOM.building_bloks.basic_residual_block import BasicResidualBlock
from MFTIQ.UOM.heads.baseline_head import HeadAbstract
import einops

class UncertaintyHead(HeadAbstract):
    def __init__(self, in_channels, n_features, n_thresholds=5, single_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_features = n_features
        self.in_channels = in_channels
        self.num_thresholds = n_thresholds
        self.single_logits = single_logits
        self.with_combination_weight = kwargs.get('with_combination_weight', False)

        self.out_channels = 1 if self.single_logits else 2

        if kwargs.get('do_not_initialize', False):
            return

        self.occlusion_head = self.define_occlusion_head_net(in_channels=self.in_channels, n_features=self.n_features,
                                                             out_channels=self.out_channels)

        self.uncertainty_head_dict = nn.ModuleDict()
        for n_threshold in range(1, n_thresholds+1):
            self.uncertainty_head_dict[f'uncertainty{n_threshold}'] = self.define_uncertainty_head_net(
                in_channels=self.in_channels, n_features=self.n_features, out_channels=self.out_channels)


    def forward(self, x, *args, **kwargs):
        occlusion_est = self.occlusion_head(x)
        outputs = {'occlusion': occlusion_est}

        for n_threshold in range(1, self.num_thresholds + 1):
            key = f'uncertainty{n_threshold}'
            outputs[key] = self.uncertainty_head_dict[key](x)

        return outputs

    @torch.inference_mode()
    def inference_outputs_formatter(self, outputs, *args, **kwargs):
        inference_occlusion_mode = kwargs.get('inference_occlusion_mode', 'with_uncertainty')
        x = outputs['occlusion']
        H, W = x.shape[2], x.shape[3]

        C = 1 if self.single_logits else 2
        assert outputs['occlusion'].shape == (1, C, H, W)
        for i in range(1, self.num_thresholds + 1):
            assert outputs[f'uncertainty{i}'].shape == (1, C, H, W), f'uncertainty{i}'

        for i in range(1, self.num_thresholds + 1):
            key = f'uncertainty{i}'
            if self.single_logits:
                outputs[key] = F.sigmoid(outputs[key])[0]
            else:
                outputs[key] = outputs[key].softmax(dim=1)[0, 1:, :, :]

        if self.single_logits:
            outputs['occlusion'] = F.sigmoid(outputs['occlusion'])[0]
        else:
            outputs['occlusion'] = outputs['occlusion'].softmax(dim=1)[0, 1:, :, :]

        not_matched = torch.ones_like(outputs['occlusion'], dtype=torch.bool)
        if inference_occlusion_mode == 'with_uncertainty':
            for i in range(1, self.num_thresholds + 1):
                not_matched = torch.logical_and(not_matched, outputs[f'uncertainty{i}'] > 0.5)
        elif inference_occlusion_mode == 'without_uncertainty':
            print("OCCL WITHOUT UNCERTAINTY")
            not_matched = torch.zeros_like(outputs['occlusion'], dtype=torch.bool)
        else:
            raise NotImplementedError

        if kwargs.get('binary_uncertainty_construction', False):
            uncertainty = torch.zeros_like(outputs['occlusion'], dtype=torch.float32)
            for i, thr in enumerate(range(1, self.num_thresholds + 1)):
                ## over_thr_mask = outputs[f'uncertainty{i}'] >= 0.5
                # e.g. when over all thresholds, the score will be: 1*1 + 1*2 + 1*4 = 7
                # e.g. when over thresholds 1 and 5, the score will be: 1*1 + 0*2 + 1*4 = 5
                # of course the outputs are not strictly binary 0, 1, but soft score between 0 and 1

                # when we sort by uncertainty, we will get lexicographic sort on the (>5, >3, >1) softmax scores
                uncertainty += 2 ** i * outputs[f'uncertainty{thr}']
            uncertainty = torch.clip(uncertainty, min=0)  # shouldn't be needed, just a safeguard
        else:
            uncertainty = 100.0 * torch.ones_like(outputs['occlusion'], dtype=torch.float32)
            for i in reversed(range(1, self.num_thresholds + 1)):
                under_thr_mask = outputs[f'uncertainty{i}'] <= 0.5
                uncertainty[under_thr_mask] = (i - 1) + (outputs[f'uncertainty{i}'][under_thr_mask] * 2.0)

            uncertainty = torch.sqrt(torch.clip(uncertainty, 0.0, 1000.0))

        if self.with_combination_weight:
            for k in outputs.keys():
                if 'combination_weight' in k:
                    outputs[k] = einops.rearrange(outputs[k], '1 C H W -> C H W')



        # Augmented outputs:
        # occlusion gives occlusion value after softmax, or 1 (is occluded) if all uncertainty thresholds
        # are greater than 0.5 (after softmax)
        # uncertainty gives artificial uncertainty value according to thresholds (look to the code)
        outputs['occlusion'][not_matched] = 1.0
        outputs['uncertainty'] = uncertainty

        assert outputs['occlusion'].shape == (1, H, W)
        assert outputs['uncertainty'].shape == (1, H, W)
        return outputs


class UncertaintyStackHead(UncertaintyHead):
    def __init__(self, in_channels, n_features, n_thresholds=5, single_logits=False,
                 with_combination_weight=False,
                 *args, **kwargs):

        super().__init__(in_channels=in_channels, n_features=n_features,
                         n_thresholds=n_thresholds, single_logits=single_logits,
                         with_combination_weight=with_combination_weight,
                         *args, do_not_initialize=True, **kwargs)

        self.occlusion_head = self.define_occlusion_head_net(in_channels=self.in_channels,
                                                             n_features=self.n_features, out_channels=self.out_channels)

        self.uncertainty_head = self.define_uncertainty_head_net(in_channels=self.in_channels,
                                                                 n_features=self.n_features,
                                                                 out_channels=(self.out_channels * n_thresholds))

        if self.with_combination_weight:
            self.combination_weight_head = self.define_base_head_net(
                in_channels=self.in_channels + 2 * self.out_channels * (n_thresholds + 1),
                n_features=self.n_features, out_channels=1
            )


    def define_base_head_net(self, in_channels, n_features=1, out_channels=2):
        return nn.Sequential(
            BasicResidualBlock(in_channels, n_features, padding=1, kernel_size=3),
            BasicResidualBlock(n_features, n_features, padding=1, kernel_size=3),
            BasicBlock(n_features, out_channels, padding=1, kernel_size=3, relu=False, use_group_norm=False),
        )


    def forward(self, x, *args, **kwargs):
        occlusion_est = self.occlusion_head(x)
        outputs = {'occlusion': occlusion_est}

        stack_unc = self.uncertainty_head(x)
        list_unc = torch.split(stack_unc, self.out_channels, dim=1)
        for idx, n_threshold in enumerate(range(1, self.num_thresholds + 1)):
            key = f'uncertainty{n_threshold}'
            outputs[key] = list_unc[idx]

        if self.with_combination_weight:
            cw_input = torch.cat([x,
                                  occlusion_est.detach(),
                                  F.sigmoid(occlusion_est).detach(),
                                  stack_unc.detach(),
                                  F.sigmoid(stack_unc).detach()], dim=1)
            outputs['combination_weight'] = self.combination_weight_head(cw_input)

        return outputs
