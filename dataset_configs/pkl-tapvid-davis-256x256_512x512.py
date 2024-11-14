from pathlib import Path
from MFTIQ.config import Config


def get_config():
    conf = Config()
    conf.pickles = ['datasets/tapvid_davis.pkl']
    conf.scaling = '256x256_512x512'
    conf.name = Path(__file__).stem
    return conf
