from pathlib import Path
from MFTIQ.config import Config
from MFTIQ.flowformer import FlowFormerPPWrapper
from MFTIQ.FlowFormerPlusPlus.configs.submissions import get_cfg


def get_config():
    conf = Config()

    conf.of_class = FlowFormerPPWrapper
    conf_name = Path(__file__).stem

    conf.flowformer_cfg = get_cfg()
    conf.flowformer_cfg.model = 'checkpoints/flowformerpp-sintel.pth'

    conf.flow_cache_dir = Path(f'flow_cache/{conf_name}/')
    conf.flow_cache_ext = '.flowouX16.pkl'
    conf.name = Path(__file__).stem

    return conf
