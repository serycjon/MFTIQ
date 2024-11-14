from pathlib import Path
from MFTIQ.config import Config

from MFTIQ.ptlflow_adapter import PTLFlowWrapper
from MFTIQ import code_path

def get_config():
    conf = Config()

    conf.of_class = PTLFlowWrapper
    conf_name = Path(__file__).stem

    conf.model_name = 'gmflow'
    conf.checkpoint_name = 'sintel'

    conf.flow_cache_dir = Path(f'flow_cache/{conf_name}/')
    conf.flow_cache_ext = '.flowouX16.pkl'
    conf.name = conf_name

    return conf
