import copy

from rich import print
from rich.panel import Panel
from rich.console import Console
from rich.text import Text

from MFTIQ.UOM.utils.attribute_dict import AttrDict
from MFTIQ.RAFT.core.raft import RAFT  # noqa: E402
from MFTIQ.UOM.network.uom_baseline4 import UOMNetBase4

import logging
logger = logging.getLogger(__name__)

class UOMConfigs:
    def __init__(self, args):
        self.args = args
        if isinstance(args, dict):
            self.args = AttrDict(**args)
        self.config_name = self.args.config
        configs = {
            'mftiq': UOMConfigBaseline4BCE200k(),
            'bs4_200k': UOMConfigBaseline4BCE200k(),

            # add own configs here
        }
        try:
            self.config = configs[self.config_name]
        except KeyError:
            logger.error(f"Unknown config name '{self.config_name}'. Should be one of {list(configs.keys())}")
            raise

        self.model = self.config.model
        self.flow_method = self.config.flow_method

    def get_flow_method(self):
        return self.flow_method

    def get_model(self):
        return self.model

    def get_flow_args(self):
        return self.config.get_flow_args(self.args)

    def get_uom_args(self):
        return self.config.get_uom_args(self.args)

    def get_train_args(self):
        return self.config.get_training_args(self.args)

    def print_config(self):
        console = Console()

        console.print(f'[blue]UOM config name:  [bold red]{self.config_name}')
        console.print(f'[blue]UOM config class: [bold green]{self.config.__class__.__name__}')

        self.print_box(self.args, f'[bold]CONSOLE ARGS', 'magenta')
        self.print_box(self.get_flow_args(), f'[bold]FLOW ARGS - [bold red]{self.flow_method.__name__}', 'blue')
        self.print_box(self.get_train_args(), f'[bold]TRAINING ARGS [bold red]{self.config.__class__.__name__}', 'yellow')
        self.print_box(self.get_uom_args(), f'[bold]UOM ARGS [bold red]{self.model.__name__}', 'green')

    def print_box(self, x, title=None, border_style=None):
        console = Console()
        with console.capture() as capture:
            console.print(x)
        str_output = capture.get()
        text = Text.from_ansi(str_output)
        console.print(Panel(text, title=title, title_align='left',
                            border_style=border_style))

# MFTIQ - WACV2025
class UOMConfigBaseline:
    def __init__(self):
        self.model = None
        self.flow_method = RAFT

    def get_flow_args(self, args):
        args_dict = {
            'occlusion_module': 'separate_with_uncertainty',
            'small': False,
            'mixed_precision': False,
            'checkpoint': 'checkpoints/raft-things-sintel-kubric-splitted-occlusion-uncertainty-non-occluded-base-sintel.pth',
        }
        return AttrDict(**args_dict)

    def get_uom_args(self, args):
        args_dict = {
            'uom_features': 16,
            'uom_use_group_norm': False,
            'cost_volume_displacement': None,
            'exp_forgetting_alphas': None,
        }
        return AttrDict(**args_dict)

    def get_training_args(self, args):
        flow_args = self.get_flow_args(args)
        args_dict = {
            'occlusion_module': flow_args.occlusion_module,
            'small': flow_args.small,
            'mixed_precision': flow_args.mixed_precision,
            'iters': 12,
            'epsilon': 1e-8,
            'clip': 1.0,
            'dropout': 0.0,
            'add_noise': False,
            'dashcam_augmenentation': False,
            'blend_source': False,
            'normalized_features': False,
            'seed': 1234,
            'stage': 'sintel_things_kubric_train_subsplit',
            'validation': ['sintel_val_subsplit', 'kubric2024_val_subsplit', 'kubric_val_subsplit'],
            'freeze_optical_flow_training': True,
            'uncertainty_loss': 'huber_non_occluded',
            'occlusion_loss': 'cross_entropy',
            'freeze_features_training': True,
            'num_steps': 50000,
            'batch_size': args.batch_size,
            'exp_forgetting_alphas': None,
            'load_occl_valid_masks': True,
            'lr': 0.000250,
            'image_size': [368, 768],
            'wdecay': 0.00001,
            'gamma': 0.85,
            'weighting_unc_loss': True,
            # 'optical_flow_loss': None,
            'flow_loss': None,
            'random_pick_flow': False,
            'augmentation_rotate90': False,
            'recurrent': None,
            'restore_checkpoint': getattr(args, 'restore_ckpt', None),
            'n_workers': getattr(args, 'n_workers', 8)
        }
        return AttrDict(**args_dict)


# ====================================================================
# =                                                                  =
# =                            BASELINE 4                            =
# =                                                                  =
# ====================================================================

# MFTIQ - WACV2025
class UOMConfigBaseline4BCE(UOMConfigBaseline):
    def __init__(self):
        super().__init__()
        self.model = UOMNetBase4
        self.flow_method = RAFT

    def get_uom_args(self, args):
        attrdict = copy.deepcopy(super().get_uom_args(args))
        attrdict.uom_features = 32
        attrdict.uom_use_group_norm = True
        attrdict.single_logits = True

        attrdict.downsample_coef = 8
        attrdict.head_type = 'uncthreshold_stack'
        attrdict.with_spatial_correlation_sampler = True
        attrdict.upsample_type = 'learned'

        attrdict.with_resnet = True
        attrdict.with_imgcnn = True
        attrdict.with_dinov2 = True
        attrdict.with_vggnet = False

        return attrdict

    def get_training_args(self, args):
        attrdict = copy.deepcopy(super().get_training_args(args))
        attrdict.uncertainty_loss = 'binary_cross_entropy'
        attrdict.occlusion_loss = 'binary_cross_entropy'
        attrdict.stage = 'kubric_long_train_subsplit'
        attrdict.random_pick_flow = True
        attrdict.num_steps = 50000
        attrdict.lr = 0.00250
        return attrdict

# MFTIQ - WACV2025
class UOMConfigBaseline4BCE200k(UOMConfigBaseline4BCE):
    def __init__(self):
        super().__init__()

    def get_uom_args(self, args):
        attrdict = copy.deepcopy(super().get_uom_args(args))

        attrdict.uom_features = 32
        attrdict.downsample_coef = 4
        attrdict.upsample_type = 'bilinear'
        attrdict.with_spatial_correlation_sampler = True

        return attrdict

    def get_training_args(self, args):
        attrdict = copy.deepcopy(super().get_training_args(args))
        attrdict.num_steps = 200000
        return attrdict
