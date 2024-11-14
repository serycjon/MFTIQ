# -*- origami-fold-style: triple-braces; coding: utf-8; -*-
import einops
import numpy as np
import torch
from types import SimpleNamespace
import logging

from MFTIQ.results import FlowOUTrackingResult
from MFTIQ.utils.io import get_flowou_with_cache, FeatureCache
from MFTIQ.utils.timing import general_time_measurer
from MFTIQ.uom import UOM, uom_cache_name

logger = logging.getLogger(__name__)


class MFTIQ():
    def __init__(self, config):
        """Create MFTIQ tracker
        args:
          config: a MFTIQ.config.Config, for example from configs/MFTIQ4_ROMA_200k_cfg.py"""
        self.C = config   # must be named self.C, will be monkeypatched!
        self.device = 'cuda'
        self.flower = config.flow_config.of_class(config.flow_config)  # init the OF

        # independent (on the flow) uncertainty+occlusion module:
        self.uom = UOM(config.uom_config)

        self.flow_timer = general_time_measurer('flow', cuda_sync=True, start_now=False, active=False,
                                                skip_warmup=15)

        self.gt_queries = None

    def init(self, img, start_frame_i=0, time_direction=1,
             flow_cache=None, feat_cache=None,
             keep_on_gpu=False, **kwargs):
        """Initialize MFTIQ on first frame

        args:
          img: opencv image (numpy uint8 HxWxC array with B, G, R channel order)
          start_frame_i: [optional] init frame number (used for caching)
          time_direction: [optional] forward = +1, or backward = -1 (used for caching)
          flow_cache: [optional] MFTIQ.utils.io.FlowCache (for caching OF on GPU, RAM, or SSD)
          feat_cache: [optional] MFTIQ.utils.io.FeatureCache (for caching UOM features on GPU or SSD)
          kwargs: [unused] - for compatibility with other trackers

        returns:
          meta: initial frame result container, with initial (zero-motion) MFTIQ.results.FlowOUTrackingResult in meta.result 
        """
        self.img_H, self.img_W = img.shape[:2]
        self.start_frame_i = start_frame_i
        self.current_frame_i = self.start_frame_i
        assert time_direction in [+1, -1]
        self.time_direction = time_direction
        self.flow_cache = flow_cache
        self.feat_cache = feat_cache

        if feat_cache is None:
            self.feat_cache = FeatureCache(0)

        self.memory = {
            self.start_frame_i: {
                'img': img,
                'result': FlowOUTrackingResult.identity((self.img_H, self.img_W), device=self.device)
            }
        }

        self.template_img = img.copy()

        self.back_memory = {}
        meta = SimpleNamespace()
        meta.result = self.memory[self.start_frame_i]['result'].clone()

        if not keep_on_gpu:
            meta.result = meta.result.cpu()

        self.gt_queries = kwargs.get('gt_queries', None)  # for oracle experiments: (N_queries xy)
        self.gt_data = kwargs.get('gt_data', None)

        return meta

    @torch.inference_mode()
    def track(self, input_img, debug=False, keep_on_gpu=False, **kwargs):
        """Track one frame

        args:
          input_img: opencv image (numpy uint8 HxWxC array with B, G, R channel order)
          debug: [optional] enable debug visualizations
          kwargs: [unused] - for compatibility with other trackers

        returns:
          meta: current frame result container, with MFTIQ.results.FlowOUTrackingResult in meta.result
                The meta.result represents the accumulated flow field from the init frame, to the current frame
        """
        meta = SimpleNamespace()
        self.current_frame_i += self.time_direction

        # OF(init, t) candidates using different deltas
        delta_results = {}
        already_used_left_ids = []
        chain_timer = general_time_measurer('chain', cuda_sync=True, start_now=False, active=self.C.timers_enabled)
        ou_timer = general_time_measurer('UOM', cuda_sync=True, start_now=False,
                                         active=self.C.timers_enabled)
        for delta in self.C.deltas:
            # candidates are chained from previous result (init -> t-delta) and flow (t-delta -> t)
            # when tracking backward, the chain consists of previous result (init -> t+delta) and flow(t+delta -> t)
            left_id = self.current_frame_i - delta * self.time_direction
            right_id = self.current_frame_i

            # we must ensure that left_id is not behind the init frame
            if self._is_before_start(left_id):
                if np.isinf(delta):
                    left_id = self.start_frame_i
                else:
                    continue
            left_id = int(left_id)

            # because of this, different deltas can result in the same left_id, right_id combination
            # let's not recompute the same candidate multiple times
            if left_id in already_used_left_ids:
                continue

            left_img = self.memory[left_id]['img']
            right_img = input_img

            template_to_left = self.memory[left_id]['result']

            flow_init = None
            use_cache = np.isfinite(delta) or self.C.cache_delta_infinity
            left_to_right = get_flowou_with_cache(self.flower, left_img, right_img, flow_init,
                                                  self.flow_cache, left_id, right_id,
                                                  read_cache=use_cache, write_cache=use_cache,
                                                  flow_timer=self.flow_timer)

            chain_timer.start()
            template_to_right = chain_results(template_to_left, left_to_right)
            already_used_left_ids.append(left_id)
            chain_timer.stop()

            ou_timer.start()
            template_img = self.memory[self.start_frame_i]['img']
            uom_name = uom_cache_name(self.uom)
            left_id = self.start_frame_i
            right_id = self.current_frame_i
            left_cache = self.feat_cache.read(left_id, uom_name)
            right_cache = self.feat_cache.read(right_id, uom_name)
            uo_outputs = self.uom(template_img, right_img, template_to_right.flow,
                                  left_cache=left_cache, right_cache=right_cache)
            self.feat_cache.write(left_id, uo_outputs['left_cache'], uom_name)
            self.feat_cache.write(right_id, uo_outputs['right_cache'], uom_name)
            template_to_right.occlusion = uo_outputs['occlusion']
            template_to_right.sigma = uo_outputs['uncertainty']
            ou_timer.stop()

            delta_results[delta] = template_to_right

        ou_timer.report('mean')
        ou_timer.report('sum')

        chain_timer.report('mean')
        chain_timer.report('sum')


        selection_timer = general_time_measurer('selection', cuda_sync=True, start_now=True,
                                                active=self.C.timers_enabled)

        def sorting_fce(key):
            if np.isinf(key):
                return 0
            else:
                return key

        used_deltas = sorted(list(delta_results.keys()), key=lambda delta: sorting_fce(delta))
        all_results = [delta_results[delta] for delta in used_deltas]
        all_flows = torch.stack([result.flow for result in all_results], dim=0)  # (N_delta, xy, H, W)
        all_sigmas = torch.stack([result.sigma for result in all_results], dim=0)  # (N_delta, 1, H, W)
        all_occlusions = torch.stack([result.occlusion for result in all_results], dim=0)  # (N_delta, 1, H, W)

        scores = -all_sigmas
        occlusion_score_penalty = float('inf')
        if self.C.occlusion_score_penalty:
            occlusion_score_penalty = self.C.occlusion_score_penalty
        scores[all_occlusions > self.C.occlusion_threshold] -= occlusion_score_penalty


        best = scores.max(dim=0, keepdim=True)
        selected_delta_i = best.indices  # (1, 1, H, W)

        best_flow = all_flows.gather(dim=0,
                                     index=einops.repeat(selected_delta_i,
                                                         'N_delta 1 H W -> N_delta xy H W',
                                                         xy=2, H=self.img_H, W=self.img_W))
        best_occlusions = all_occlusions.gather(dim=0, index=selected_delta_i)
        best_sigmas = all_sigmas.gather(dim=0, index=selected_delta_i)
        selected_flow, selected_occlusion, selected_sigmas = best_flow, best_occlusions, best_sigmas

        selected_flow = einops.rearrange(selected_flow, '1 xy H W -> xy H W', xy=2, H=self.img_H, W=self.img_W)
        selected_occlusion = einops.rearrange(selected_occlusion, '1 1 H W -> 1 H W', H=self.img_H, W=self.img_W)
        selected_sigmas = einops.rearrange(selected_sigmas, '1 1 H W -> 1 H W', H=self.img_H, W=self.img_W)

        result = FlowOUTrackingResult(selected_flow, selected_occlusion, selected_sigmas)

        # mark flows pointing outside of the current image as occluded
        invalid_mask = einops.rearrange(result.invalid_mask(), 'H W -> 1 H W')
        result.occlusion[invalid_mask] = 1
        selection_timer.report()

        out_result = result.clone()
        if self.C.out_occlusion_threshold:
            out_result.occlusion[out_result.occlusion > self.C.out_occlusion_threshold] = 1
            out_result.occlusion[out_result.occlusion <= self.C.out_occlusion_threshold] = 0

        if kwargs.get('return_selected_deltas', False):
            sane_deltas = [abs(self.current_frame_i - self.start_frame_i) if isinstance(delta, str) or np.isinf(delta) else delta
                           for delta in used_deltas]
            selected_deltas = einops.rearrange(torch.from_numpy(np.array(sane_deltas))[selected_delta_i.cpu()],
                                               '1 1 H W -> 1 H W')
            meta.selected_deltas = selected_deltas

        meta.result = out_result
        if not keep_on_gpu:
            meta.result = meta.result.cpu()

        self.memory[self.current_frame_i] = {'img': input_img,
                                             'result': result}

        self._cleanup_memory()
        self.flow_timer.report('mean')
        return meta

    # @profile
    def _cleanup_memory(self):
        # max delta, ignoring the inf special case
        try:
            max_delta = np.amax(np.array(self.C.deltas)[np.isfinite(self.C.deltas)])
        except ValueError:  # only direct flow
            max_delta = 0
        memory_frames = list(self.memory.keys())
        for mem_frame_i in memory_frames:
            if mem_frame_i == self.start_frame_i:
                continue

            if self.time_direction > 0 and mem_frame_i + max_delta > self.current_frame_i:
                # time direction     ------------>
                # mem_frame_i ........ current_frame_i ........ (mem_frame_i + max_delta)
                # ... will be needed later
                continue

            if self.time_direction < 0 and mem_frame_i - max_delta < self.current_frame_i:
                # time direction     <------------
                # (mem_frame_i - max_delta) ........ current_frame_i .......... mem_frame_i
                # ... will be needed later
                continue

            del self.memory[mem_frame_i]

    def _is_before_start(self, frame_i):
        return ((self.time_direction > 0 and frame_i < self.start_frame_i) or  # forward
                (self.time_direction < 0 and frame_i > self.start_frame_i))    # backward


def chain_results(left_result, right_result):
    flow = left_result.chain(right_result.flow)
    return FlowOUTrackingResult(flow, None, None)
