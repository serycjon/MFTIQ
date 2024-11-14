# -*- origami-fold-style: triple-braces; coding: utf-8; -*-
import sys
from types import SimpleNamespace
import logging
from itertools import product

import einops
import numpy as np
import cv2
import torch
from ipdb import iex

from MFTIQ.results import FlowOUTrackingResult
from MFTIQ.utils.io import get_flowou_with_cache
from MFTIQ.utils.geom_utils import torch_get_featuremap_coords
from MFTIQ.utils.timing import general_time_measurer
from MFTIQ.utils.misc import ensure_numpy, dummy_profile
from MFTIQ.uom import UOM
import MFTIQ.utils.vis_utils as vu
from MFTIQ.utils.io import FeatureCache

logger = logging.getLogger(__name__)
profile = dummy_profile()

class MFT():
    def __init__(self, config):
        """Create MFT tracker
        args:
          config: a MFT.config.Config, for example from configs/MFT_cfg.py"""
        self.C = config   # must be named self.C, will be monkeypatched!
        self.flower = config.flow_config.of_class(config.flow_config)  # init the OF
        if config.uom_config:
            self.uom = UOM(config.uom_config)
        self.device = 'cuda'
        self.flow_timer = general_time_measurer('flow', cuda_sync=True, start_now=False, active=False)

        self.img_H, self.img_W = None, None
        self.start_frame_i = 0
        self.current_frame_i = 0
        self.time_direction = 1
        self.memory = None

        self.back_memory = None

        self.flow_cache = None
        self.feat_cache = None

        self.gt_queries = None

    def init(self, img, start_frame_i=0, time_direction=1,
             flow_cache=None, feat_cache=None,
             keep_on_gpu=False, **kwargs):
        """Initialize MFT on first frame

        args:
          img: opencv image (numpy uint8 HxWxC array with B, G, R channel order)
          start_frame_i: [optional] init frame number (used for caching)
          time_direction: [optional] forward = +1, or backward = -1 (used for caching)
          flow_cache: [optional] MFT.utils.io.FlowCache (for caching OF on GPU, RAM, or SSD)
          feat_cache: [optional] MFT.utils.io.FeatureCache (for caching UOM features on GPU or SSD)
          kwargs: for compatibility with other trackers and debugging stuff, oracle experiment

        returns:
          meta: initial frame result container, with initial (zero-motion) MFT.results.FlowOUTrackingResult in meta.result 
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

        self.back_memory = {}
        meta = SimpleNamespace()
        meta.result = self.memory[self.start_frame_i]['result'].clone()
        if not keep_on_gpu:
            meta.result = meta.result.cpu()

        self.gt_queries = kwargs.get('gt_queries', None)  # for oracle experiments: (N_queries xy)

        return meta

    @profile
    def track(self, input_img, debug=False, keep_on_gpu=False, **kwargs):
        """Track one frame

        args:
          input_img: opencv image (numpy uint8 HxWxC array with B, G, R channel order)
          debug: [optional] enable debug visualizations
          kwargs: [unused] - for compatibility with other trackers

        returns:
          meta: current frame result container, with MFT.results.FlowOUTrackingResult in meta.result
                The meta.result represents the accumulated flow field from the init frame, to the current frame
        """
        meta = SimpleNamespace()
        self.current_frame_i += self.time_direction

        # OF(init, t) candidates using different deltas
        delta_results = {}
        already_used_left_ids = []
        chain_timer = general_time_measurer('chain', cuda_sync=True, start_now=False, active=self.C.timers_enabled)
        uom_timer = general_time_measurer('UOM', cuda_sync=True, start_now=False, active=self.C.timers_enabled)
        for delta in self.C.deltas:
            # candidates are chained from previous result (init -> t-delta) and flow (t-delta -> t)
            # when tracking backward, the chain consists of previous result (init -> t+delta) and flow(t+delta -> t)
            left_id = self.current_frame_i - delta * self.time_direction
            right_id = self.current_frame_i

            # we must ensure that left_id is not behind the init frame
            if self.is_before_start(left_id):
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
            if self.C.flow_init:
                # init the flow with the coordinates from the previous frame {{{
                # example:
                # we are at frame T, doing delta=8 -> we need to initialize OF(T-8, T) with OF(T-8, T-1)
                # to do that, we create a meshgrid "in the T-8 coord system" and backwarp it to frame 0
                # after that, we can just forward warp the coordinates to T-1 using known OF(0, T-1)
                # finally, we subtract the original meshgrid to convert from coordinates to flow

                coords_left = torch_get_featuremap_coords((input_img.shape[0], input_img.shape[1]),  # (xy, H, W)
                                                          device=self.device, keep_shape=True).to(torch.float32)
                coords_in_template = template_to_left.warp_backward(coords_left)

                last_result = self.memory[self.current_frame_i - self.time_direction]['result']
                coords_last = einops.rearrange(
                    last_result.warp_forward_points(
                        einops.rearrange(coords_in_template, 'xy H W -> (H W) xy', xy=2)),
                    '(H W) xy -> xy H W', **einops.parse_shape(coords_in_template, 'xy H W'))
                flow_init = coords_last - coords_left
                # }}}

            use_cache = np.isfinite(delta) or self.C.cache_delta_infinity
            left_to_right = get_flowou_with_cache(self.flower, left_img, right_img, flow_init,
                                                  self.flow_cache, left_id, right_id,
                                                  read_cache=use_cache, write_cache=use_cache,
                                                  flow_timer=self.flow_timer)
            if self.C.uom_config:
                uom_timer.start()
                uom_name = self.uom.C.config_name
                left_cache = self.feat_cache.read(left_id, uom_name)
                right_cache = self.feat_cache.read(right_id, uom_name)
                uo_outputs = self.uom(left_img, right_img, left_to_right.flow,
                                      left_cache=left_cache, right_cache=right_cache)
                self.feat_cache.write(left_id, uo_outputs['left_cache'], uom_name)
                self.feat_cache.write(right_id, uo_outputs['right_cache'], uom_name)

                left_to_right.occlusion = uo_outputs['occlusion']
                left_to_right.sigma = uo_outputs['uncertainty']
                uom_timer.stop()

            chain_timer.start()
            delta_results[delta] = chain_results(template_to_left, left_to_right)
            already_used_left_ids.append(left_id)
            chain_timer.stop()

        uom_timer.report('mean')
        uom_timer.report('sum')
        chain_timer.report('mean')
        chain_timer.report('sum')

        if self.C.backcheck:
            # update the back_memory
            # ----------------------
            # back_memory[i] contains simple-chained flow from #(t-1) to #i
            # we will update it to contain flow from #t (current frame) to #i
            # we will also throw away the result for the smallest 'i' (if too far in history),
            # and add a new "chain" directly from #t to #(t-1)
            right_id = self.current_frame_i - self.time_direction
            if not self.is_before_start(right_id):
                last_back_flow = get_flowou_with_cache(
                    self.flower, left_img=input_img, right_img=self.memory[right_id]['img'],
                    flow_init=None, cache=self.flow_cache,
                    left_id=self.current_frame_i, right_id=right_id,
                    read_cache=True, write_cache=True)
                timer = general_time_measurer('back_memory_update', active=self.C.timers_enabled)
                update_back_memory(self.back_memory, last_back_flow, self.current_frame_i, self.time_direction,
                                   self.C.max_back_chain_len)
                timer.report()

        selection_timer = general_time_measurer('selection', cuda_sync=True, start_now=True,
                                                active=self.C.timers_enabled)
        used_deltas = sorted(list(delta_results.keys()), key=lambda delta: 0 if np.isinf(delta) else delta)
        all_results = [delta_results[delta] for delta in used_deltas]
        all_flows = torch.stack([result.flow for result in all_results], dim=0)  # (N_delta, xy, H, W)
        all_sigmas = torch.stack([result.sigma for result in all_results], dim=0)  # (N_delta, 1, H, W)
        all_occlusions = torch.stack([result.occlusion for result in all_results], dim=0)  # (N_delta, 1, H, W)

        scores = -all_sigmas
        if self.C.random_sort:
            scores = torch.rand_like(scores)
        occlusion_score_penalty = float('inf')
        if self.C.occlusion_score_penalty:
            occlusion_score_penalty = self.C.occlusion_score_penalty
        scores[all_occlusions > self.C.occlusion_threshold] -= occlusion_score_penalty

        if self.C.oracle_selection:
            gt_queries = self.gt_queries  # (N_queries xy)
            gt_pos = torch.from_numpy(kwargs['gt_pos']).to(self.device)  # (N_queries, xy)
            gt_occl = kwargs['gt_occl']  # (N_queries)
            oracle_scores, oracle_occlusions = select_by_oracle(all_flows, gt_queries, gt_pos, gt_occl, scores, all_occlusions)

            if self.C.oracle_scores:
                scores = oracle_scores

            if self.C.oracle_occlusion:
                all_occlusions = oracle_occlusions

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
        if not self.C.oracle_occlusion:  # do not override the oracle... it is pretty smart you know...
            result.occlusion[invalid_mask] = 1
        selection_timer.report()

        # backcheck!
        if self.C.backcheck:
            backcheck_timer = general_time_measurer('backcheck', cuda_sync=True, active=self.C.timers_enabled)
            invalid_mask = backcheck(result, self.back_memory, self.memory,
                                     self.current_frame_i, self.time_direction,
                                     self.C.max_back_chain_len, self.C.backcheck_max_epe_sq,
                                     input_img, self.memory[self.current_frame_i]['img'], debug=debug)
            result.occlusion[invalid_mask] = 1
            backcheck_timer.report()

        out_result = result.clone()
        if self.C.out_occlusion_threshold:
            out_result.occlusion[out_result.occlusion > self.C.out_occlusion_threshold] = 1
            out_result.occlusion[out_result.occlusion <= self.C.out_occlusion_threshold] = 0

        if kwargs.get('return_selected_deltas', False):
            sane_deltas = [abs(self.current_frame_i - self.start_frame_i) if np.isinf(delta) else delta
                           for delta in used_deltas]
            selected_deltas = einops.rearrange(torch.from_numpy(np.array(sane_deltas))[selected_delta_i.cpu()],
                                               '1 1 H W -> 1 H W').to(torch.float16)
            meta.selected_deltas = selected_deltas

        meta.result = out_result
        if not keep_on_gpu:
            meta.result = meta.result.cpu()

        self.memory[self.current_frame_i] = {'img': input_img,
                                             'result': result}

        self.cleanup_memory()
        self.flow_timer.report('mean')
        return meta

    # @profile
    def cleanup_memory(self):
        """Delete no-longer-needed entries from result memory

        i.e. beyond max delta from the current frame

        Also compress old but still relevant entries, if configured with C.compress_memory"""
        # max delta, ignoring the inf special case
        try:
            max_delta = np.amax(np.array(self.C.deltas)[np.isfinite(self.C.deltas)])
        except ValueError:  # only direct flow
            max_delta = 0
        has_direct_flow = np.any(np.isinf(self.C.deltas))
        memory_frames = list(self.memory.keys())
        total_space = 0
        total_count = 0
        compressed_count = 0
        for mem_frame_i in memory_frames:
            res = self.memory[mem_frame_i]['result']
            total_count += 1

            if res.compressed():
                compressed_count += 1
                total_count -= 1
            if res.device == 'cuda':
                total_space += res.used_memory()

            if mem_frame_i == self.start_frame_i and has_direct_flow:
                continue

            frame_dist = abs(mem_frame_i - self.current_frame_i)
            if (self.C.compress_memory and frame_dist > self.C.compress_memory and
                not self.memory[mem_frame_i]['result'].compressed()):

                logger.debug(f'Compressing memory @ frame {mem_frame_i}')
                self.memory[mem_frame_i]['result'].compress()

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
        logger.debug((f"@#{self.current_frame_i}: Total GPU RAM occupied by the MFT memory: "
                      f"{total_space / 1024**3:0.1f}GB (in {total_count} FlowOUs). {compressed_count} compressed."))

    def is_before_start(self, frame_i):
        """Return whether a given frame_i is before (or after when tracking backward) starting frame"""
        return ((self.time_direction > 0 and frame_i < self.start_frame_i) or  # forward
                (self.time_direction < 0 and frame_i > self.start_frame_i))    # backward


def update_back_memory(back_memory, last_back_flow, current_frame_i, time_direction, max_back_chain_len):
    r"""Update back_memory (for backcheck)

    back_memory stores flows from (t-1) to (t-2), (t-3), ..., (t-max_back_chain_len)
    last_back_flow stores flow from t to (t-1)
    we need to reconnect / chain the flows
    and throw away the last one (t-max_back_chain_len)
    the picture is for +1 time direction
          /----------------------\
         /      /-----------------\
        /      /       /-----------\
       /      /       /       /-----\   /----\
      /      /       /       /       \ /      \
    t-max  t-4     t-3     t-2       t-1       t
    """

    # delete the oldest
    if (current_frame_i - time_direction * max_back_chain_len) in back_memory:
        del back_memory[current_frame_i - time_direction * max_back_chain_len]

    # then replace one-by-one with the chain from frame t
    frame_ids = range(
        current_frame_i - time_direction * (max_back_chain_len - 1),  # oldest ignored
        current_frame_i - time_direction,  # newest result currently into t-2
        time_direction)
    for frame_i in frame_ids:
        if frame_i not in back_memory:
            continue
        # originally flow from (t-1) to 'i'
        back_memory[frame_i] = chain_results(last_back_flow, back_memory[frame_i])
        # now from 't' to 'i'

    # add the last flow (into t-1)
    back_memory[current_frame_i - time_direction] = last_back_flow


def backcheck(result, back_memory, memory, current_frame_i, time_direction,
              max_back_chain_len, backcheck_max_epe_sq,
              cur_img, template_img, debug):
    """Perform backcheck

    track backwards with simple flow chain to filter out incorrect re-detections
    """
    occl_thr = 0.5
    device = result.flow.device
    invalid_mask = torch.zeros((1, result.flow.shape[1], result.flow.shape[2]), dtype=torch.bool, device=device)
    valid_mask = torch.zeros((1, result.flow.shape[1], result.flow.shape[2]), dtype=torch.bool, device=device)
    redetection_mask = memory[current_frame_i - time_direction]['result'].occlusion > occl_thr
    dbg_invalid_reason = torch.zeros((1, result.flow.shape[1], result.flow.shape[2]), dtype=torch.uint8, device=device)

    frame_ids = range(current_frame_i - time_direction,
                      current_frame_i - time_direction * (max_back_chain_len + 1),  # include the last one
                      -time_direction)
    for enum_i, frame_i in enumerate(frame_ids):
        if frame_i not in back_memory:
            continue
        forward_backward = chain_results(result, back_memory[frame_i])
        from_memory = memory[frame_i]['result']
        # 'from_memory' is the tracking result on a previous frame 'i'
        # forward_backward is made by chaining the current frame result with a short backward chain
        # In order to pass the backcheck, the forward_backward must either end in occlusion / outside image
        # or it must stay close to the 'from_memory' flow all the time
        FB_distance_sq = einops.reduce(torch.square(forward_backward.flow - from_memory.flow),
                                       'xy H W -> 1 H W', xy=2, reduction='sum')

        # not marked as invalid so far, now the back-chain becomes occluded or out-of-view (= .invalid_mask())
        # => this one is valid, whatever happens next in the backchaining
        valid_mask[~invalid_mask & ((forward_backward.occlusion > occl_thr) | (forward_backward.invalid_mask()))] = True

        # both 'from_memory' and 'FB_distance_sq' are visible, so they should have the same positions
        invalid_mask[(FB_distance_sq > backcheck_max_epe_sq) & (from_memory.occlusion < occl_thr) &
                     (forward_backward.occlusion < occl_thr)] = True
        dbg_invalid_reason[(dbg_invalid_reason == 0) &
                           (FB_distance_sq > backcheck_max_epe_sq) &
                           (from_memory.occlusion < occl_thr) &
                           (forward_backward.occlusion < occl_thr)] = enum_i + 1

    # the ones that were visible for 'max_back_chain_len' frames do not come from a disocclusion -> invalid re-detection,
    # but only if the 'from_memory' was occluded.  Otherwise we should decide only by the FB_distance_sq
    invalid_mask[(forward_backward.occlusion < occl_thr) & (from_memory.occlusion > occl_thr)] = True
    dbg_invalid_reason[(dbg_invalid_reason == 0) &
                       (forward_backward.occlusion < occl_thr) &
                       (from_memory.occlusion > occl_thr)] = 255

    # finally, unmark invalid_mask, where we already have valid_mask
    invalid_mask[valid_mask] = False
    invalid_mask[~redetection_mask] = False
    dbg_invalid_reason[~invalid_mask] = 0

    dbg_invalid_reason = einops.rearrange(ensure_numpy(dbg_invalid_reason), '1 H W -> H W')
    dbg_invalid_reason = 255 * (dbg_invalid_reason > 0).astype(np.uint8)
    dbg_invalid_reason = vu.cv2_colormap(dbg_invalid_reason, do_colorbar=False)

    if debug:
        # sample random points from invalid_mask
        # N_pts = 10
        # try:
        #     vis_pts = gu.sample_coords_from_mask(einops.rearrange(ensure_numpy(invalid_mask), '1 H W -> H W'),
        #                                          N=N_pts)  # (xy, N)
        # except Exception:
        #     vis_pts = np.zeros((2, 0))

        @iex
        def handler(event, x, y, _flags, _param):
            # cmap = vu.get_cmap('tab20')
            vis = vu.to_gray_3ch(cur_img)
            vis_pts = np.array([[x, y]], dtype=np.float32)
            cur_pts = ensure_numpy(result.warp_forward_points(vis_pts))
            vu.circle(vis, cur_pts[0, :].tolist(), 2, vu.WHITE, thickness=-1)

            cur_fb_pts = cur_pts.copy()
            # N_pts_real = 1
            for frame_i in frame_ids:
                if frame_i not in back_memory:
                    continue
                forward_backward = chain_results(result, back_memory[frame_i])
                from_memory = memory[frame_i]['result']

                pts_fb = ensure_numpy(forward_backward.warp_forward_points(vis_pts))
                pts_mem = ensure_numpy(from_memory.warp_forward_points(vis_pts))
                occl_fb = ensure_numpy(forward_backward.sample(vis_pts)[1])
                occl_mem = ensure_numpy(from_memory.sample(vis_pts)[1])
                # img = memory[frame_i]['img']
                # vis = vu.to_gray_3ch(img)
                i = 0
                # vu.line(vis, pts_fb[i, :].tolist(), pts_mem[i, :].tolist(), (0, 0, 0), thickness=1)
                # color = vu.cv2_cmap_get(cmap, 2 * i).tolist()
                color = vu.RED
                if occl_fb[0, i] > occl_thr:
                    color = (0, 0, 60)
                vu.line(vis, cur_fb_pts[i, :].tolist(), pts_fb[i, :].tolist(), color, thickness=1)
                vu.circle(vis, pts_fb[i, :].tolist(), 2, color, thickness=-1)
                # color = vu.cv2_cmap_get(cmap, 2 * i + 1).tolist()
                color = vu.GREEN
                if occl_mem[0, i] > occl_thr:
                    color = (0, 60, 0)
                vu.line(vis, cur_pts[i, :].tolist(), pts_mem[i, :].tolist(), color, thickness=1)
                vu.circle(vis, pts_mem[i, :].tolist(), 2, color, thickness=-1)
                cur_pts = pts_mem
                cur_fb_pts = pts_fb
            if not handler.history_active:
                vu.imshow("cv: vis_trajs", vis)

            if event == cv2.EVENT_LBUTTONDOWN:
                handler.history_active = True
                vis_frame_i = current_frame_i
                while True:
                    if vis_frame_i == current_frame_i:
                        vis = vu.to_gray_3ch(cur_img)
                    else:
                        vis = vu.to_gray_3ch(memory[vis_frame_i]['img'])

                    if vis_frame_i == current_frame_i:
                        forward_backward = result
                        from_memory = result

                        pts_fb = ensure_numpy(forward_backward.warp_forward_points(vis_pts))
                        pts_mem = ensure_numpy(from_memory.warp_forward_points(vis_pts))
                        occl_fb = ensure_numpy(forward_backward.sample(vis_pts)[1])
                        occl_mem = ensure_numpy(from_memory.sample(vis_pts)[1])
                    else:
                        forward_backward = chain_results(result, back_memory[vis_frame_i])
                        from_memory = memory[vis_frame_i]['result']

                        pts_fb = ensure_numpy(forward_backward.warp_forward_points(vis_pts))
                        pts_mem = ensure_numpy(from_memory.warp_forward_points(vis_pts))
                        occl_fb = ensure_numpy(forward_backward.sample(vis_pts)[1])
                        occl_mem = ensure_numpy(from_memory.sample(vis_pts)[1])
                    color = vu.RED
                    i = 0
                    diff_sq = einops.reduce(np.square(pts_fb - pts_mem),
                                            'N xy -> N', xy=2, reduction='sum')
                    diff_state = 'OK'
                    if diff_sq > backcheck_max_epe_sq:
                        diff_state = 'big'
                    backcheck_state = 'visible'
                    if occl_fb[0, i] > occl_thr:
                        color = (0, 0, 60)
                        backcheck_state = 'occluded'
                    vu.circle(vis, pts_fb[i, :].tolist(), 2, color, thickness=-1)
                    color = vu.GREEN
                    hist_state = 'visible'
                    if occl_mem[0, i] > occl_thr:
                        color = (0, 60, 0)
                        hist_state = 'occluded'
                    vu.circle(vis, pts_mem[i, :].tolist(), 2, color, thickness=-1)
                    vis = vu.draw_text(
                        vis,
                        (f'EPE: {diff_state} | ', 'mem', f': {hist_state} | ', 'back', f': {backcheck_state}'),
                        color=(vu.WHITE, vu.GREEN, vu.WHITE, vu.RED, vu.WHITE),
                        pos='tl', size=1, thickness=1)

                    vu.imshow("cv: history", vis)
                    c = cv2.waitKey(0)
                    if c == ord('q'):
                        sys.exit(1)
                    elif c == 13:
                        vu.imshow("cv: history", np.zeros_like(vis))
                        cv2.waitKey(5)
                        handler.history_active = False
                        break
                    elif c in [ord('h'), 81]:  # left
                        if (vis_frame_i - 1) in back_memory or (vis_frame_i - 1 == current_frame_i):
                            vis_frame_i -= 1
                    elif c in [ord('l'), 83]:  # right
                        if (vis_frame_i + 1) in back_memory or (vis_frame_i + 1 == current_frame_i):
                            vis_frame_i += 1

        handler.history_active = False

        alpha = 0.5
        vis = cv2.addWeighted(dbg_invalid_reason, alpha,
                              vu.to_gray_3ch(template_img), 1 - alpha, 0)
        vu.imshow("cv: vis", vis)
        vu.imshow("cv: vis_trajs", vu.to_gray_3ch(cur_img))
        cv2.setMouseCallback("cv: vis", handler)

        # vu.imshow("cv: dbg_invalid_reason", dbg_invalid_reason)
        while True:
            c = cv2.waitKey(20)
            if c == ord('q'):
                sys.exit(1)
            if c == ord(' '):
                break
    return invalid_mask


@profile
def chain_results(left_result, right_result):
    """Chain two FlowOUTrackingResults, O = max(O1, O2), U = U1 + U2"""
    flow = left_result.chain(right_result.flow)
    occlusions = torch.maximum(left_result.occlusion,
                               left_result.warp_backward(right_result.occlusion))
    sigmas = torch.sqrt(torch.square(left_result.sigma) +
                        torch.square(left_result.warp_backward(right_result.sigma)))
    return FlowOUTrackingResult(flow, occlusions, sigmas)

@profile
def select_by_oracle(flows, gt_queries, gt_pos, gt_occl,
                     scores, occlusions):
    """Select the best flow candidates at GT query positions (around them due to interpolation)

    Since the GT queries are not on the grid (non-integer coordinates), the final MFT output will be created by bilinear interpolation.
    The oracle thus needs to select the best flow in all the 4 neighboring grid points.
    We pretend that each gt_query has different neighboring grid points (optimization too crazy otherwise).

    args:
        flows: (N_delta, xy, H, W) tensor
        gt_queries: (N_queries, xy) tensor
        gt_pos: (N_queries, xy) tensor
        gt_occl: (N_queries, ) tensor
        scores: (N_delta, 1, H, W) tensor with original scores (will be copied and updated)
        occlusions: (N_delta, 1, H, W) tensor with original occlusion scores (will be copied and updated)

    returns:
        scores: (N_delta, 1, H, W) updated scores
        occlusions: (N_delta, 1, H, W) updated scores
    """
    scores = scores.clone()
    occlusions = occlusions.clone()

    device = flows.device

    N_queries = gt_queries.shape[0]
    N_deltas = flows.shape[0]
    H, W = flows.shape[2], flows.shape[3]

    x, y = gt_queries[:, 0], gt_queries[:, 1]

    ## pre-compute bilinear coefficients
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    # border=repeat (flow chaining uses zero border, but hopefully it doesn't matter much)
    x0 = torch.clamp(x0, 0, W - 1)
    x1 = torch.clamp(x1, 0, W - 1)
    y0 = torch.clamp(y0, 0, H - 1)
    y1 = torch.clamp(y1, 0, H - 1)

    x0f = x0.float()
    x1f = x1.float()
    y0f = y0.float()
    y1f = y1.float()

    w_a = einops.rearrange((x1f - x) * (y1f - y), 'N_queries -> 1 N_queries 1', N_queries=N_queries)
    w_b = einops.rearrange((x1f - x) * (y - y0f), 'N_queries -> 1 N_queries 1', N_queries=N_queries)
    w_c = einops.rearrange((x - x0f) * (y1f - y), 'N_queries -> 1 N_queries 1', N_queries=N_queries)
    w_d = einops.rearrange((x - x0f) * (y - y0f), 'N_queries -> 1 N_queries 1', N_queries=N_queries)

    # construct all the combinations of flow selections in all 4 nearest neighbors
    # this could be extracted to a function with memoization...
    selections = torch.from_numpy(np.array(list(product(range(N_deltas), repeat=4)))).to(device)
    # selections: (N_selections 4) tensor

    # sample all the combinations in all the 4 neighbors at the same time
    data_a = flows[einops.rearrange(selections[:, 0], 'N -> N 1'), :, einops.rearrange(y0, 'M -> 1 M'), einops.rearrange(x0, 'M -> 1 M')]
    data_b = flows[einops.rearrange(selections[:, 1], 'N -> N 1'), :, einops.rearrange(y1, 'M -> 1 M'), einops.rearrange(x0, 'M -> 1 M')]
    data_c = flows[einops.rearrange(selections[:, 2], 'N -> N 1'), :, einops.rearrange(y0, 'M -> 1 M'), einops.rearrange(x1, 'M -> 1 M')]
    data_d = flows[einops.rearrange(selections[:, 3], 'N -> N 1'), :, einops.rearrange(y1, 'M -> 1 M'), einops.rearrange(x1, 'M -> 1 M')]
    # data_*: (N_selections N_queries xy)

    assert (data_a[0, 0, :] == flows[selections[0, 0], :, y0[0], x0[0]]).all()
    assert (data_b[0, 0, :] == flows[selections[0, 1], :, y1[0], x0[0]]).all()
    if N_queries > 3:
        assert (data_a[0, 2, :] == flows[selections[0, 0], :, y0[2], x0[2]]).all()
        assert (data_c[0, 3, :] == flows[selections[0, 2], :, y0[3], x1[3]]).all()
    if N_deltas > 1:
        assert (data_a[1, 0, :] == flows[selections[1, 0], :, y0[0], x0[0]]).all()
        assert (data_b[1, 0, :] == flows[selections[1, 1], :, y1[0], x0[0]]).all()

    interp_flow = (w_a * data_a) + (w_b * data_b) + (w_c * data_c) + (w_d * data_d)  # (N_selections N_queries xy)
    interp_pos = interp_flow + einops.rearrange(gt_queries, 'N_queries xy -> 1 N_queries xy', N_queries=N_queries, xy=2)

    diff = interp_pos - einops.rearrange(gt_pos, 'N_queries xy -> 1 N_queries xy', N_queries=N_queries, xy=2)
    dist = einops.reduce(torch.square(diff),
                         'N_selections N_queries xy -> N_selections N_queries',
                         reduction='sum',
                         N_selections=N_deltas**4, N_queries=N_queries)

    best = dist.min(dim=0)
    best_selection_i = best.indices  # (N_queries)

    best_selection = selections[best_selection_i, :]  # (N_queries 4)

    # set infinite scores for the selected delta ids
    # TODO: set random selection if all the dists are too big?
    scores[best_selection[:, 0], 0, y0, x0] = torch.inf
    scores[best_selection[:, 1], 0, y1, x0] = torch.inf
    scores[best_selection[:, 2], 0, y0, x1] = torch.inf
    scores[best_selection[:, 3], 0, y1, x1] = torch.inf

    # set occlusions occlusions
    # TODO: set occlusion if all the dists are too big?
    torch_gt_occl = torch.from_numpy(gt_occl).to(device).float()

    ## does not work when only the oracle occlusions are used (candidate selection without oracle)
    # occlusions[best_selection[:, 0], 0, y0, x0] = torch_gt_occl
    # occlusions[best_selection[:, 1], 0, y1, x0] = torch_gt_occl
    # occlusions[best_selection[:, 2], 0, y0, x1] = torch_gt_occl
    # occlusions[best_selection[:, 3], 0, y1, x1] = torch_gt_occl

    ## replace occlusion in all deltas
    occlusions[:, 0, y0, x0] = torch_gt_occl
    occlusions[:, 0, y1, x0] = torch_gt_occl
    occlusions[:, 0, y0, x1] = torch_gt_occl
    occlusions[:, 0, y1, x1] = torch_gt_occl

    return scores, occlusions
