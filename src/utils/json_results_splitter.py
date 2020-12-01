#!/usr/bin/env python3

""" Split a JSON Results (and Metrics) file into separated files. """

import argparse
import cProfile
import io
import json
import os
import pstats
import sys
import traceback

from pprint import pformat, pprint
from tqdm import tqdm

def argument_parser():
    """ Argument parser """
    parser = argparse.ArgumentParser(
        description='Split a JSON Results (and Metrics) file into separated files.')
    parser.add_argument(
        '--input', required=True, type=str, help='INPUT file.')
    parser.add_argument(
        '--output-dir', required=True, type=str,
        help='Path to the directory to save the files.')
    parser.add_argument(
        '--profiler', dest='profiler', action='store_true',
        help='Enables cProfile.')
    parser.set_defaults(profiler=False)
    return parser.parse_args()

ARGS = argument_parser()

def _main():
    """ Splitting loop """
    with open(ARGS.input, 'r') as jsonfile:
        for row in tqdm(jsonfile): # enumerate cannot be used due to the size of the file
            result = json.loads(row)
            metric_file = os.path.join(
                ARGS.output_dir, 'metrics_{}.json'.format(result['training_iteration']))
            with open(metric_file, 'w') as fstream:
                json.dump(result, fstream)

if __name__ == '__main__':
    ret = 0
    ## ========================              PROFILER              ======================== ##
    if ARGS.profiler:
        profiler = cProfile.Profile()
        profiler.enable()
    ## ========================              PROFILER              ======================== ##
    try:
        _main()
    except Exception: # traci.exceptions.TraCIException: libsumo.libsumo.TraCIException:
        ret = 666
        EXC_TYPE, EXC_VALUE, EXC_TRACEBACK = sys.exc_info()
        traceback.print_exception(EXC_TYPE, EXC_VALUE, EXC_TRACEBACK, file=sys.stdout)
    finally:
        ## ========================          PROFILER              ======================== ##
        if ARGS.profiler:
            profiler.disable()
            results = io.StringIO()
            pstats.Stats(profiler, stream=results).sort_stats('cumulative').print_stats(50)
            print('Profiler: \n%s', results.getvalue())
        ## ========================          PROFILER              ======================== ##
        sys.exit(ret)
