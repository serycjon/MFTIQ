from MFTIQ.MFT import MFT
from pathlib import Path
from MFTIQ.config import Config, load_config
import numpy as np

import logging
logger = logging.getLogger(__name__)


def get_config():
    conf = Config()

    conf.tracker_class = MFT
    conf.flow_config = load_config('configs/flow/RAFTou_kubric_huber_split_nonoccl.py')
    conf.deltas = [np.inf, 1, 2, 4, 8, 16, 32]
    conf.occlusion_threshold = 0.02
    conf.cache_delta_infinity = True

    conf.name = Path(__file__).stem
    return conf
