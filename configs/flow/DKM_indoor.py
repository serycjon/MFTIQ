from pathlib import Path
from MFTIQ.config import Config
from MFTIQ.dkm import DKMWrapper
# from MFTIQ.DKM.dkm import DKMv3_outdoor
from MFTIQ.DKM.dkm import DKMv3_indoor


def get_config():
    conf = Config()

    conf.of_class = DKMWrapper
    conf_name = Path(__file__).stem

    conf.model = DKMv3_indoor

    conf.flow_cache_dir = Path(f'flow_cache/{conf_name}/')
    conf.flow_cache_ext = '.flowouX16.pkl'
    conf.name = Path(__file__).stem

    return conf
