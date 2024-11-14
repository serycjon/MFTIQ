from pathlib import Path
from MFTIQ.config import Config

def get_config():
    conf = Config()
    kinetics_dir = Path('datasets/tapvid_kinetics/')
    conf.pickles = sorted(list(kinetics_dir.glob('*.pkl')))  # {shard_i:04d}_of_0100.pkl'
    conf.scaling = '256x256_512x512'
    conf.name = Path(__file__).stem
    return conf
