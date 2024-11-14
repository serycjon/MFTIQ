# mypy: disable-error-code="attr-defined"
from MFTIQ import MFTIQ
from MFTIQ.UOM.network.configs import UOMConfigs

from pathlib import Path
from MFTIQ.config import Config, load_config
import numpy as np

import logging
logger = logging.getLogger(__name__)


def get_config():
    conf = Config()

    conf.tracker_class = MFTIQ
    conf.flow_config = load_config('configs/flow/RoMa_outdoor.py')
    conf.deltas = [np.inf, 1, 2, 4, 8, 16, 32]
    conf.occlusion_threshold = 0.5
    conf.cache_delta_infinity = True
    conf.occlusion_score_penalty = 10000

    conf.uom_config = UOMConfigs({'config': 'bs4_200k'})
    conf.uom_config.checkpoint = 'checkpoints/UOM_bs4_200k.pth'
    conf.uom_config.binary_uncertainty_construction = True

    conf.timers_enabled = False

    conf.name = Path(__file__).stem

    # unused
    conf.out_occlusion_threshold = False

    return conf
