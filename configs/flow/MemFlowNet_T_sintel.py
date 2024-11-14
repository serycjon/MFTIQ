from pathlib import Path
from MFTIQ.config import Config

from MFTIQ.memflow import MemFlowWrapper
from MFTIQ.MemFlow.core.Networks import build_network
from MFTIQ.MemFlow.configs.sintel_memflownet_t import get_cfg

from MFTIQ import code_path
def get_config():
    conf = Config()

    conf.of_class = MemFlowWrapper
    conf_name = Path(__file__).stem

    # TODO: Continue HERE
    conf.model = build_network
    conf.memflow_weights = 'checkpoints/MemFlowNet_T_sintel.pth'
    conf.memflow_params = get_cfg()

    conf.flow_cache_dir = Path(f'flow_cache/{conf_name}/')
    conf.flow_cache_ext = '.flowouX16.pkl'
    conf.name = Path(__file__).stem

    return conf
