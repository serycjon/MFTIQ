# -*- origami-fold-style: triple-braces; coding: utf-8; -*-
import logging
import re
import sys
import traceback
import warnings
from pathlib import Path

import pandas as pd
from tabulate import tabulate
from ipdb import iex
from rich.console import Console

from MFTIQ.runners.eval_MFT_tapvid import run as run_evaluation
from MFTIQ.runners.run_MFT_tapvid import parse_arguments, get_parser
from MFTIQ.runners.run_MFT_tapvid import run as run_tracker

colorful = Console()

pd.set_option('display.precision', 1)
logger = logging.getLogger(__name__)


SUBSET = ['bike-packing', 'soapbox', 'camel']

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))
    log.write('\n\n')


def method_rename(config_name):
    # config_name = re.sub(r"^MPT_multiflow_occl_sigmasq_occlinvalid", "MFT", config_name)
    config_name = re.sub(r"_cfg$", "", config_name)
    return config_name

@iex
def run(args):
    if not args.skip_run:
        try:
            run_tracker(args)
        except Exception:
            logger.exception("Tracking failed")
            raise
    if not args.skip_eval:
        run_evaluation(args)
    if not args.skip_report:
        report(args)

    return 0


def report(args):
    if args.mode in ['first', 'both']:
        print('FIRST:')
        report_first(args)
    if args.mode in ['strided', 'both']:
        print('\n\nSTRIDED:')
        report_strided(args)

def report_first(args):
    return report_aux(args, 'tapvid-eval.pklz')

def report_strided(args):
    return report_aux(args, 'tapvid-eval-strided.pklz')

def report_aux(args, pickle_name):
    all_methods = []
    res_dirs = [args.export] if args.extra_dirs is None else [args.export, *args.extra_dirs]
    paths = []
    n_incomplete_results = 0
    for extra_i, res_dir in enumerate(res_dirs):
        paths = sorted(list(res_dir.glob(f'*/eval/{pickle_name}')))
        for path in sorted(paths):
            method_name = method_rename(path.parent.parent.stem)
            if extra_i > 0:
                method_name = f"{method_name} [{extra_i}]"

            method_df = pd.read_pickle(path)

            # print(method_df)
            try:
                expected_n_sequences = [30, # tapvid davis
                                        265, # robotap
                                        ]
                if args.subset:
                    expected_n_sequences = len(SUBSET)
                    method_df = method_df.loc[method_df['seq'].isin(SUBSET)]

                if len(method_df) not in expected_n_sequences:
                    logger.debug(f'{method_name} was skipped. It does not contain one of expected number of sequences ({expected_n_sequences}), but {len(method_df)}.')
                    n_incomplete_results += 1
                    # continue
                    pass
                method_results = method_df[['average_prec', 'average_pts_within_thresh',
                                            'pts_within_1', 'pts_within_2', 'pts_within_4',
                                            'pts_within_8', 'pts_within_16', 'occlusion_accuracy',
                                            'average_jaccard']].mean() * 100
                method_results['vis_prec'] = (method_df['occlusion_TN'] / (method_df['occlusion_TN'] + method_df['occlusion_FN'])).mean() * 100
                method_results['vis_recall'] = (method_df['occlusion_TN'] / (method_df['occlusion_TN'] + method_df['occlusion_FP'])).mean() * 100
            except KeyError:
                continue
            method_results['method'] = method_name
            # method_results['resolution'] = resolution
            method_results = method_results.to_frame().T

            all_methods.append(method_results)

    if n_incomplete_results > 0 and not args.verbose:
        logger.warning(f"Skipped {n_incomplete_results} experiments: didn't have results on all the sequences (run with -v to get a full list)")
    results = pd.concat(all_methods)
    # assuming TAP-Vid DAVIS
    if 'strided' in pickle_name and not args.subset: # STRIDED
        results = pd.concat([results, pd.DataFrame([{'average_pts_within_thresh': 53.1, 'occlusion_accuracy': 82.3, 'average_jaccard': 38.4,
                                  'method': 'TAP-Net', 'resolution': '256'}])], ignore_index=True)
        results = pd.concat([results, pd.DataFrame([{'average_pts_within_thresh': 59.4, 'occlusion_accuracy': 82.1, 'average_jaccard': 42.0,
                                  'method': 'PIPs', 'resolution': '256'}])], ignore_index=True)
        results = pd.concat([results, pd.DataFrame([{'average_pts_within_thresh': 67.5, 'occlusion_accuracy': 85.3, 'average_jaccard': 51.7,
                                  'method': 'OmniMotion', 'resolution': '256'}])], ignore_index=True)
        results = pd.concat([results, pd.DataFrame([{'average_pts_within_thresh': 72.3, 'occlusion_accuracy': 87.6, 'average_jaccard': 61.3,
                                  'method': 'TAPIR', 'resolution': '256'}])], ignore_index=True)
        results = pd.concat([results, pd.DataFrame([{'average_pts_within_thresh': 79.1, 'occlusion_accuracy': 88.7, 'average_jaccard': 64.8,
                                  'method': 'CoTracker', 'resolution': '256'}])], ignore_index=True)
        results = pd.concat([results, pd.DataFrame([{'average_pts_within_thresh': 78.5, 'occlusion_accuracy': 90.7, 'average_jaccard': 66.4,
                                 'method': 'BootsTAP', 'resolution': '256'}])], ignore_index=True)
        results = pd.concat([results, pd.DataFrame([{'average_pts_within_thresh': 79.2, 'occlusion_accuracy': 91.0, 'average_jaccard': 66.3,
                                 'method': 'TAPTR', 'resolution': '256'}])], ignore_index=True)
    elif not args.subset: # FIRST
        results = pd.concat([results, pd.DataFrame([{'average_pts_within_thresh': 48.6, 'occlusion_accuracy': 78.8, 'average_jaccard': 33.0,
                                  'method': 'TAP-Net', 'resolution': '256'}])], ignore_index=True)
        results = pd.concat([results, pd.DataFrame([{'average_pts_within_thresh': 70.0, 'occlusion_accuracy': 86.5, 'average_jaccard': 56.2,
                                  'method': 'TAPIR', 'resolution': '256'}])], ignore_index=True)
        results = pd.concat([results, pd.DataFrame([{'average_pts_within_thresh': 75.4, 'occlusion_accuracy': 89.3, 'average_jaccard': 60.6,
                                  'method': 'CoTracker', 'resolution': '256'}])], ignore_index=True)
        results = pd.concat([results, pd.DataFrame([{'average_pts_within_thresh': 74.0, 'occlusion_accuracy': 88.4, 'average_jaccard': 61.4,
                                  'method': 'BootsTAP', 'resolution': '256'}])], ignore_index=True)
        results = pd.concat([results, pd.DataFrame([{'average_pts_within_thresh': 76.1, 'occlusion_accuracy': 91.1, 'average_jaccard': 63.0,
                                  'method': 'TAPTR', 'resolution': '256'}])], ignore_index=True)


    results['cfg'] = results['method']
    # results['method'] = results['method'].apply(method_rename)
    first_column = results.pop('method')
    results.insert(0, 'method', first_column)

    results = results.rename(columns={'average_pts_within_thresh': '< thrs',
                                      'occlusion_accuracy': 'OA',
                                      'average_jaccard': 'AJ',
                                      'pts_within_1': '< 1',
                                      'pts_within_2': '< 2',
                                      'pts_within_4': '< 4',
                                      'pts_within_8': '< 8',
                                      'pts_within_16': '< 16'})
    results = results[['method', 'AJ', '< thrs', 'OA', 'vis_prec', 'vis_recall', '< 1', '< 2', '< 4', '< 8', '< 16']]

    if args.methods is not None:
        optional_extra_link_re = r'( \[\d+\])?'
        def get_sort_i(method):
            for sort_i, allowed in enumerate(args.methods):
                if re.match(f'{allowed}{optional_extra_link_re}$', method):
                    return sort_i
            return float('nan')
        results['sort_i'] = results.method.apply(get_sort_i)
        results = results[~results['sort_i'].isna()]
        results = results.sort_values(by=['sort_i', 'method'])
        del results['sort_i']
        results = results.reset_index(drop=True)

    if args.return_results:
        return results

    print()
    table = tabulate(results, headers="keys", tablefmt=args.table_format, floatfmt=".2f")
    for idx, line in enumerate(table.split('\n')):
        if ' MFT ' in line:
            colorful.print(f'[bold blue]{line}', highlight=False)
        elif idx % 2 == 0:
            colorful.print(f'[bold]{line}', highlight=False)
        else:
            colorful.print(f'{line}', highlight=False)

    for extra_i, res_dir in enumerate(res_dirs):
        def uses_extra(extra_i):
            def aux(method):
                return re.match(r'.* \[' + str(extra_i) + r'\]$', method)
            return aux
        extra_i_used = results.method.apply(uses_extra(extra_i)).any()
        if extra_i_used:
            print(f'[{extra_i}] {res_dir.resolve()}')

    print()
    if args.subset:
        colorful.print(f'[yellow]WARNING: EVALUATED ONLY ON A SEQUENCE SUBSET: {", ".join(SUBSET)}[/yellow]')


def main():
    parser = get_parser()
    parser.add_argument('--skip_run', help='', action='store_true')
    parser.add_argument('--skip_eval', help='', action='store_true')
    parser.add_argument('--skip_report', help='', action='store_true')
    parser.add_argument('--extra_dirs', help='More export directories (only for report printing)', type=Path, nargs='+')
    parser.add_argument('--table_format', default='orgtbl', type=str, help='Format of python-tabulate output')
    parser.add_argument('--subset', help='report on smaller subset', action='store_true')
    parser.add_argument('--methods', help='selected methods to report', nargs='+')
    parser.add_argument('--return_results', action='store_true', help='return results instead of printing')
    args = parse_arguments(parser)

    if args.subset:
        args.seq = SUBSET

    if args.verbose:
        warnings.showwarning = warn_with_traceback

    return run(args)


if __name__ == '__main__':
    sys.exit(main())
