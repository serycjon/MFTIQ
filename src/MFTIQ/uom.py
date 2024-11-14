import torch
import einops
import logging
import warnings
from pathlib import Path

from MFTIQ.results import FlowOUTrackingResult

logger = logging.getLogger(__name__)


class UOM(object):
    def __init__(self, config):
        self.device = 'cuda'
        self.C = config

        with warnings.catch_warnings():
            # catch warning from DINOv2 because guys from
            # facebookresearch are misusing warnings to inform about
            # success...
            warnings.filterwarnings(action='ignore',
                                    category=UserWarning,
                                    message=r'xFormers is available.*')
            uom = torch.nn.DataParallel(config.model(**config.get_uom_args()))
        uom = uom.module
        if config.checkpoint != 'dummy':
            uom.load_state_dict(torch.load(config.checkpoint, map_location='cpu'))
        uom.requires_grad_(False)
        uom.to(self.device)
        uom.eval()
        self.uom = uom

    def __call__(self, left_img, right_img, flow,
                 left_cache=None, right_cache=None,
                 history=None, rotation=0):
        """Compute occlusion and uncertainty

        args:
            left_img: (H W BGR) uint8 numpy array of left image
            right_img: (H W BGR) uint8 numpy array of right image
            flow: (xy H W) float cuda tensor with optical flow from left to right image
            left_cache: [optional] a dict with cached features of left_image
            right_cache: [optional] a dict with cached features of right_image
            history: [optional] a motion / occlusion state history dict
                     keys: exponential forgetting coefficients (1 = use only the newest value)
                     history[coefficient]: dict with:
                         'position' (xy H W) tensor with average flow from template to previous frame
                         'velocity' (xy H W) tensor with average flow between consecutive frames
                         'occlusion' (1 H W) tensor with average occlusion
        returns:
            outputs: a dict with keys 'occlusion': (1 H W) and 'uncertainty': (1 H W)
                     and possibly other outputs, like 'left_cache' and 'right_cache' features
        """
        if 'image' in left_cache:
            image1 = left_cache['image']
        else:
            image1 = einops.rearrange(torch.from_numpy(left_img[:, :, ::-1].copy()),
                                      'H W C -> 1 C H W', C=3)
            image1 = image1.cuda().float()

        if 'image' in right_cache:
            image2 = right_cache['image']
        else:
            image2 = einops.rearrange(torch.from_numpy(right_img[:, :, ::-1].copy()),
                                      'H W C -> 1 C H W', C=3)
            image2 = image2.cuda().float()

        H, W = image1.shape[2], image1.shape[3]

        if history is not None:
            for coefficient in history.keys():
                assert history[coefficient]['position'].shape == (2, H, W)
                assert history[coefficient]['velocity'].shape == (2, H, W)
                assert history[coefficient]['occlusion'].shape == (1, H, W)

        if isinstance(flow, dict):
            # this part of "if" is for the UOM with multiple hypotheses in the input
            flow_hypotheses = {}
            for c_k, c_v in flow.items():
                if isinstance(c_v, dict) and 'flow' in c_v:
                    c_flow = c_v['flow']
                elif isinstance(c_v, FlowOUTrackingResult):
                    c_flow = c_v.flow
                elif isinstance(c_v, torch.Tensor):
                    c_flow = c_v
                else:
                    raise TypeError(f'Unexpected type {type(c_v)}')
                flow_hypotheses[c_k] = einops.rearrange(c_flow, 'xy H W -> 1 xy H W', xy=2, H=H, W=W)
            flow = flow_hypotheses
        else:
            flow = einops.rearrange(flow, 'xy H W -> 1 xy H W', xy=2, H=H, W=W)

        outputs = self.uom(image1, image2, flow,
                           inference_mode=True,
                           left_cache=left_cache, right_cache=right_cache,
                           binary_uncertainty_construction=getattr(self.C, 'binary_uncertainty_construction', False),
                           inference_occlusion_mode=getattr(self.C, 'inference_occlusion_mode', 'with_uncertainty'),
                           history=history,
                           rotation=rotation)

        if 'final_predictions' in outputs:
            assert outputs['final_predictions']['occlusion'].shape == (1, H, W)
            assert outputs['final_predictions']['flow'].shape == (2, H, W)
            for k, c_outputs in outputs['individual_predictions'].items():
                assert c_outputs['occlusion'].shape == (1, H, W)
                assert c_outputs['uncertainty'].shape == (1, H, W)
        else:
            assert outputs['occlusion'].shape == (1, H, W)
            assert outputs['uncertainty'].shape == (1, H, W)

        return outputs


def uom_cache_name(uom: UOM) -> str:
    return uom.C.config_name + '---' + Path(uom.C.checkpoint).stem
    
