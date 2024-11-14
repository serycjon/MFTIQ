# -*- origami-fold-style: triple-braces -*-
import sys
import os
import argparse
import tqdm
import numpy as np
import pickle
from pathlib import Path
import socket
import datetime
import einops
import shutil
import torch

from MFTIQ.utils.various import with_debugger, SparseExceptionLogger
from MFTIQ.config import load_config
import MFTIQ.utils.vis_utils as vu
import MFTIQ.utils.io as io_utils
from MFTIQ.utils.repro import code_export
from MFTIQ.evaluation import tapvid_eval_stuff as tves
from MFTIQ.point_tracking import convert_to_point_tracking

import logging
logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter, fromfile_prefix_chars='@')

    parser.add_argument('dataset', default=Path('dataset_configs/pkl-tapvid-davis-pkl-tapvid-davis-256x256_512x512.py'),
                        help='dataset config', type=Path)
    parser.add_argument('trackers', default='MFT/configs/MFT_cfg.py', help='path to tracker configs, all must share the same flow_config', type=Path,
                        nargs='+')
    parser.add_argument('--export', default='./export', help='result export directory', type=Path)
    parser.add_argument('--cache', default='./cache', help='flow cache directory', type=Path)
    parser.add_argument('--gpu', help='cuda device')
    parser.add_argument('-c', '--cont', help='skip already computed sequences', action='store_true')
    parser.add_argument('--debug', help='track with tracker debug info', action='store_true')
    parser.add_argument('-v', '--verbose', help='', action='store_true')
    parser.add_argument('--mode', help='TAP-Vid evaluation query modes', choices=['first', 'strided', 'both'],
                        default='both')
    parser.add_argument('--write_flow', help='write flowou for the frame 0 template', action='store_true')
    parser.add_argument('-fcl', '--feat_cache_limit',
                        help='number of images of which the features are cached on GPU [default = 50 to cover up to max-delta last images]',
                        type=int, default=50)
    parser.add_argument('-rcl', '--ram_cache_limit', help='RAM cache limit in GB', type=int, default=30)
    parser.add_argument('-gcl', '--gpu_cache_limit', help='GPU cache limit in GB', type=int, default=5)
    parser.add_argument('--seq', help='sequence subset', nargs='+')
    parser.add_argument('--sequence_sample_size', type=int, default=None,
                        help='flows are evaluated on specified randomly sampled sequences')
    parser.add_argument('--sequence_sample_seed', type=int, default=42,
                        help='seed for sampling the sequences')
    parser.add_argument('--persistent_cache', action='store_true',
                        help='Cached results are stored on disk and not deleted')
    parser.add_argument('--store_deltas', help='store the selected flow deltas', action='store_true')
    parser.add_argument('--write_metas', help='write tracking metadata', action='store_true')
    parser.add_argument('--only_write_images', action='store_true',
                        help='just export the dataset as png images instead doing anything else')
    parser.add_argument('--cache_preload_workers', type=int, help='number of workers for flow cache preloading')
    parser.add_argument('--cache_preload_number', type=int, help='how much cached flows will be loaded in advance')
    return parser


def parse_arguments(parser=None):
    if parser is None:
        parser = get_parser()

    args = parser.parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    stdout_lvl = logging.DEBUG if args.verbose else logging.INFO
    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(stdout_lvl)
    log_handlers = [stdout_handler]

    config_name = args.trackers[0].stem

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    if args.export is not None:
        log_dir = Path('logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f'{stamp}__{config_name}.log'
        file_handler = logging.FileHandler(log_path)
        log_handlers.append(file_handler)

    log_format = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_format, handlers=log_handlers)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("ltr.admin.loading").setLevel(logging.ERROR)

    hostname = socket.gethostname()
    cmdline = str(' '.join(sys.argv))
    logger.info(f"cmd: {cmdline}")
    logger.info(f"start: {stamp}")
    logger.info(f"host: {hostname}")

    return args


# @profile
def run(args):
    configs = [load_config(path) for path in args.trackers]
    validate_configs(configs)

    # we will load the tracker from the first config, but we have just
    # validated that all the configs share tracker class and flow so
    # we will just monkeypatch the appropriate config into the tracker
    # at runtime. Let's hope nothing breaks :D
    config = configs[0]
    tracker_cls = config.tracker_class
    tracker = tracker_cls(config)

    dataset_conf = load_config(args.dataset)

    for config in configs:
        export_dir = args.export / config.name

        code_path = export_dir / 'code'
        code_path.mkdir(parents=True, exist_ok=True)
        code_export(code_path)
        result_dir = export_dir / 'results'
        result_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == 'both':
        query_modes = ['first', 'strided']
    else:
        query_modes = [args.mode]

    for pickle_path in tqdm.tqdm(dataset_conf.pickles, desc='pkl shards', position=0, leave=None, ascii=True):
        dataset = tves.create_tapvid_dataset(pickle_path, query_modes, dataset_conf.scaling,
                                             random_sample_size=args.sequence_sample_size,
                                             random_sample_seed=args.sequence_sample_seed)
        for seq in tqdm.tqdm(dataset, desc='sequences', position=1, leave=None, ascii=True):
            cache_dirs = []
            orig_sequence_name = seq['video_name']
            if args.seq is not None and orig_sequence_name not in args.seq:
                continue

            if dataset_conf.remove_background_masks:
                bg_mask_dir = Path(dataset_conf.remove_background_masks) / orig_sequence_name
                if not bg_mask_dir.exists():
                    logger.info(f'no background segmentation for {orig_sequence_name} - skipping')
                    continue

            video = seq["data"][query_modes[0]]["video"]  # all query_modes have the same ["video"]
            video = einops.rearrange(video, '1 N_frames H W C -> N_frames H W C', C=3)
            video = video[:, :, :, ::-1].copy()  # convert from RGB to GBR and make contiguous
            N_frames = video.shape[0]

            if dataset_conf.remove_background_masks:
                import cv2
                bg_mask_paths = list(io_utils.get_video_frames(bg_mask_dir))
                assert len(bg_mask_paths) == N_frames

                for frame_i in range(N_frames):
                    mask = bg_mask_paths[frame_i][:, :, 0]  # all 3 channels are the same anyway
                    mask = tves.resize_video(einops.rearrange(mask, 'H W -> 1 H W 1'),
                                             (video.shape[1], video.shape[2]))
                    mask = einops.rearrange(mask, '1 H W 1 -> H W')
                    if dataset_conf.keep_mask_id:  # delete everything except one object id
                        # do not check for exact match, because of mask resize (Lancozs sampling)
                        mask = np.abs(mask - dataset_conf.keep_mask_id) > 0.5
                    else:  # delete background
                        mask = mask < 0.5

                    video[frame_i, mask, :] = 0
                    if False:
                        cv2.imshow("cv: video", video[frame_i, :, :, :])
                        c = cv2.waitKey(0)
                        if c == ord('q'):
                            import sys
                            sys.exit(1)

            assert video.dtype == np.uint8

            if args.only_write_images:
                import cv2
                image_export_dir = args.export / 'dataset_images' / orig_sequence_name
                image_export_dir.mkdir(parents=True, exist_ok=True)
                for i in range(N_frames):
                    frame = video[i, :, :, :]
                    cv2.imwrite(str(image_export_dir / f'{i:05d}.png'), frame)
                continue

            ram_cache_limit_MB = args.ram_cache_limit * 1e3
            gpu_cache_limit_MB = args.gpu_cache_limit * 1e3

            flow_cache_dir = args.cache / dataset_conf.name / orig_sequence_name
            ram_flow_cache, feat_cache = create_caches(flow_cache_dir, ram_cache_limit_MB, gpu_cache_limit_MB,
                                                       feature_cache_capacity=args.feat_cache_limit,
                                                       persistent_cache=args.persistent_cache,
                                                       preload=args.cache_preload_number,
                                                       num_workers=args.cache_preload_workers,
                                                       )

            cache_dirs.append(flow_cache_dir)

            for query_mode in tqdm.tqdm(query_modes, desc='query mode', position=2, leave=None, ascii=True):
                gt_data = seq["data"][query_mode]
                gt_data['seq_name'] = orig_sequence_name
                query_points = einops.rearrange(gt_data['query_points'],
                                                '1 N_queries txy -> N_queries txy').astype(np.int64)
                start_frames = np.unique(query_points[:, 0])
                if query_mode == 'first' and args.write_flow and 0 not in start_frames:
                    raise Exception("Trying to export flowous from first frame, but 0 is not in 'start_frames'" +
                                    f"in {orig_sequence_name} in {pickle_path.stem}")
                N_queries = query_points.shape[0]
                N_frames = video.shape[0]
                xy = 2
                for tracker_config in tqdm.tqdm(configs, desc='trackers', position=3, leave=None, ascii=True):
                    tracker.C = tracker_config
                    pred_occluded = np.zeros((N_queries, N_frames))
                    pred_tracks = np.zeros((N_queries, N_frames, xy))
                    is_pure_direct = np.zeros((N_queries, N_frames))

                    export_dir = args.export / tracker_config.name
                    result_dir = export_dir / 'results'
                    seq_querymode_tracker_result_path = result_dir / f'{orig_sequence_name}-{query_mode}.pklz'
                    if args.cont and seq_querymode_tracker_result_path.exists():
                        print(
                            f'skipping {orig_sequence_name}-{query_mode} for {tracker_config.name} - already computed')
                        continue
                    # accumulate the tracker results for all the query
                    # points (and both directions if "strided" evaluation mode)
                    for start_frame in tqdm.tqdm(start_frames, desc='query frame', position=4, leave=None, ascii=True):
                        current_mask = query_points[:, 0] == start_frame
                        current_queries = query_points[current_mask, 1:]
                        current_queries = current_queries[:, ::-1].copy()  # convert to xy order, make contiguous
                        torch_current_queries = torch.from_numpy(current_queries).to('cuda')

                        directions = ['forward']
                        if query_mode == 'strided':
                            directions.append('backward')
                        for direction in directions:
                            sequence_name = f'{orig_sequence_name}--{start_frame}--{direction}'
                            if export_dir is not None and args.write_metas:
                                seq_result_dir = result_dir / sequence_name
                                seq_result_dir.mkdir(parents=True, exist_ok=True)
                                meta_path = seq_result_dir / 'meta.pklz'
                            try:
                                metas = track_sequence(tracker, video, start_frame, direction=direction,
                                                       debug=args.debug, flow_cache=ram_flow_cache,
                                                       feat_cache=feat_cache,
                                                       store_selected_deltas=args.store_deltas,
                                                       # for oracle experiments
                                                       gt_data=gt_data, gt_queries=torch_current_queries, gt_query_mask=current_mask
                                                       )
                            except KeyboardInterrupt:
                                raise
                            except Exception:
                                logger.exception(f'error in sequence {sequence_name}')
                                raise

                            frame_i_gen = range(start_frame, N_frames)
                            if direction == 'backward':
                                frame_i_gen = range(start_frame, 0 - 1, -1)
                            for frame_i in frame_i_gen:
                                meta = metas[frame_i]
                                current_coords, current_occlusions =  convert_to_point_tracking(meta.result, torch_current_queries)
                                pred_tracks[current_mask, frame_i, :] = current_coords
                                pred_occluded[current_mask, frame_i] = current_occlusions
                                if True and hasattr(meta, 'selected_deltas'):
                                    from MFTIQ.utils import interpolation
                                    current_delta = einops.rearrange(
                                        interpolation.bilinear_sample(
                                            einops.rearrange(meta.selected_deltas.float(), 'C H W -> 1 C H W', C=1),
                                            einops.rearrange(torch_current_queries.cpu(), 'N_pts xy -> 1 N_pts xy', xy=2)),
                                        'batch N_pts C -> (batch N_pts C)', batch=1, C=1)
                                    # print(current_delta)
                                    inf_delta = abs(frame_i - start_frame)
                                    is_pure_direct[current_mask, frame_i] = torch.abs(current_delta - inf_delta) < 1e-6
                                    # print(np.sum(is_pure_direct[current_mask, frame_i]))

                            # export {{{
                            if args.export is not None:
                                if any([hasattr(meta, 'vis') for meta in metas.values()]):
                                    vis_path = Path(export_dir) / 'vis'
                                    vis_path.mkdir(parents=True, exist_ok=True)
                                    writer = vu.VideoWriter(vis_path / f'{sequence_name}.mp4')
                                    for frame_i in sorted(list(metas.keys())):
                                        meta = metas[frame_i]
                                        vis = getattr(meta, 'vis', None)
                                        if vis is not None:
                                            writer.write(vis)
                                    writer.close()

                                if start_frame == 0 and query_mode == 'first' and args.write_flow:
                                    flowou_dir = Path(export_dir) / 'flowous' / orig_sequence_name
                                    flowou_dir.mkdir(parents=True, exist_ok=True)
                                    for frame_i in frame_i_gen:
                                        meta = metas[frame_i]
                                        result = meta.result
                                        flowou_path = flowou_dir / f'0--{frame_i}.flowouX16.pkl'
                                        result.write(flowou_path)

                                if args.write_metas:
                                    metas = {k: prune_meta(v, ['vis', 'result']) for k, v in metas.items()}
                                    with open(meta_path, 'wb') as fout:
                                        pickle.dump(metas, fout)
                            # }}}
                    # sequence finished for one tracker and one query modeone mode finished, save the tracklets
                    H, W = video.shape[1], video.shape[2]
                    pred_occluded = einops.rearrange(pred_occluded, 'N_queries N_frames -> 1 N_queries N_frames')
                    pred_tracks = einops.rearrange(pred_tracks,
                                                   'N_queries N_frames xy -> 1 N_queries N_frames xy', xy=2)
                    scale = einops.rearrange(np.array([256.0 / W, 256.0 / H]), 'xy -> 1 1 1 xy')
                    pred_tracks *= scale
                    assert pred_tracks.shape[0] == 1
                    assert pred_tracks.shape[3] == 2
                    assert len(pred_tracks.shape) == 4
                    tracklet_outputs = {'tracks': pred_tracks,
                                        'occluded': pred_occluded,
                                        'extra': {'is_pure_direct': is_pure_direct}}
                    with open(seq_querymode_tracker_result_path, 'wb') as fout:
                        pickle.dump(tracklet_outputs, fout)

                # sequence finished for all trackers in one query_mode
            # sequence finished for all trackers, all query modes, let's clean up
            if args.persistent_cache:
                ram_flow_cache.backup_to_disk()
            ram_flow_cache.clear()

    return 0


def create_caches(cache_dir_path, ram_cache_limit_MB, gpu_cache_limit_MB,
                  feature_cache_capacity=0, persistent_cache=False, preload=False, num_workers=0):
    # if not persistent_cache:
    #     try:
    #         shutil.rmtree(cache_dir_path)
    #     except Exception:
    #         pass
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    ram_flow_cache = io_utils.FlowCache(cache_dir_path,
                                        max_RAM_MB=ram_cache_limit_MB,
                                        max_GPU_RAM_MB=gpu_cache_limit_MB,
                                        preload=preload,
                                        num_workers=num_workers)
    feature_cache = io_utils.FeatureCache(feature_cache_capacity, path=cache_dir_path / '_feat_cache_')
    return ram_flow_cache, feature_cache


# @profile
def track_sequence(tracker, video, start_frame, direction='forward', debug=False,
                   flow_cache=None, feat_cache=None,
                   store_selected_deltas=False,
                   gt_data=None, gt_queries=None, gt_query_mask=None):
    assert direction in ['forward', 'backward']
    all_metas = {}
    sparse_logger = SparseExceptionLogger(logger)

    N_frames = video.shape[0]
    initialized = False
    frame_i_gen = range(start_frame, N_frames)
    time_direction = +1
    if direction == 'backward':
        frame_i_gen = range(start_frame, 0 - 1, -1)
        time_direction = -1

    for frame_i in frame_i_gen:
        frame = video[frame_i, :, :, :]

        if not initialized:
            initialized = True
            meta = tracker.init(frame, start_frame_i=start_frame, time_direction=time_direction,
                                flow_cache=flow_cache, feat_cache=feat_cache, gt_queries=gt_queries,
                                # for video name (for SAM2 experiment)
                                gt_data=gt_data)
        else:
            try:
                if gt_data is not None:
                    gt_pos = gt_data['target_points'][0, gt_query_mask, frame_i, :]  # (N_queries, xy)
                    gt_occl = gt_data['occluded'][0, gt_query_mask, frame_i]  # (N_queries)
                else:
                    gt_pos, gt_occl = None, None

                meta = tracker.track(frame, debug=debug, return_selected_deltas=store_selected_deltas,
                                     # for oracle experiments:
                                     gt_pos=gt_pos, gt_occl=gt_occl)
            except StopIteration:
                break
            except KeyboardInterrupt:
                raise
            except Exception as ex:
                sparse_logger("Tracker exception", ex)
                raise
        meta.frame_i = frame_i
        meta.backward = (direction == 'backward')

        all_metas[frame_i] = meta

        for update_frame_i, update_meta in getattr(meta, 'updates', {}).items():
            all_metas[update_frame_i] = update_meta
    return all_metas


def prune_meta(meta, to_prune=None):
    if to_prune is None:
        to_prune = []
    meta_keys = list(meta.__dict__.keys())
    for key in meta_keys:
        if key in to_prune:
            delattr(meta, key)

    return meta


def all_same(xs):
    return all(x == xs[0] for x in xs)


def validate_configs(configs):
    # check that all the configs share the same tracker class and optical flow
    assert all_same([c.tracker_class for c in configs])
    assert all_same([c.flow_config for c in configs])


@with_debugger
def main():
    args = parse_arguments()
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
