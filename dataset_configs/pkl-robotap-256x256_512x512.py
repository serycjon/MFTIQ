from pathlib import Path
from MFTIQ.config import Config


def get_config():
    conf = Config()
    robotap_dir = Path('datasets/robotap/')
    conf.pickles = sorted(list(robotap_dir.glob('*.pkl')))  # robotap_split{shard_i}.pkl'
    conf.scaling = '256x256_512x512'
    conf.name = Path(__file__).stem
    return conf
