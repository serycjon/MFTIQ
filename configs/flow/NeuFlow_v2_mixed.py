from pathlib import Path
from MFTIQ.config import Config
from MFTIQ.neuflow_v2 import NeuFlowWrapper
from MFTIQ.NeuFlow_v2.NeuFlow.neuflow import NeuFlow
from MFTIQ import code_path

def get_config():
    conf = Config()

    conf.of_class = NeuFlowWrapper
    conf_name = Path(__file__).stem

    conf.model = NeuFlow
    conf.neuflow_weights = code_path / 'NeuFlow_v2/neuflow_mixed.pth'

    conf.flow_cache_dir = Path(f'flow_cache/{conf_name}/')
    conf.flow_cache_ext = '.flowouX16.pkl'
    conf.name = Path(__file__).stem

    return conf
