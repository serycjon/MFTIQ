from pathlib import Path
from MFTIQ.config import Config
from MFTIQ.roma import RoMaWrapper
from MFTIQ.RoMa.roma import roma_outdoor
# from MFTIQ.RoMa.roma import roma_indoor


def get_config():
    conf = Config()

    conf.of_class = RoMaWrapper
    conf_name = Path(__file__).stem

    conf.model = roma_outdoor

    conf.flow_cache_dir = Path(f'flow_cache/{conf_name}/')
    conf.flow_cache_ext = '.flowouX16.pkl'
    conf.name = Path(__file__).stem

    return conf
