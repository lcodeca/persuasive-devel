#!/usr/bin/env python3

""" Process the RLLIB logs/result.json """

import argparse
import cProfile
import io
import logging
from pprint import pformat
import pstats

from stataggrexps import StatAggrExps

def _argument_parser():
    """ Argument parser for the stats parser. """
    parser = argparse.ArgumentParser(
        description='RLLIB & SUMO Statistics parser.')
    parser.add_argument(
        '--dir', required=True, type=str, help='Main directory.')
    parser.add_argument(
        '--pattern', required=True, type=str,
        help='Input substring to look for the JSON file in multiple folders.')
    parser.add_argument(
        '--prefix', default='stats', help='Output prefix for the processed data.')
    parser.add_argument(
        '--begin', required=True, type=int, help='Output prefix for the processed data.')
    parser.add_argument(
        '--end', required=True, type=int, help='Output prefix for the processed data.')
    parser.add_argument(
        '--profiler', dest='profiler', action='store_true', help='Enable cProfile.')
    parser.set_defaults(profiler=False)
    return parser.parse_args()

def _main():
    """ Process the RLLIB logs/result.json """

    config = _argument_parser()

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler = cProfile.Profile()
        profiler.enable()
    ## ========================              PROFILER              ======================== ##

    statistics = StatAggrExps(config.dir, config.pattern, config.prefix, config.begin, config.end)
    statistics.reward_over_agents()
    logging.info('Done')

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler.disable()
        results = io.StringIO()
        pstats.Stats(profiler, stream=results).sort_stats('cumulative').print_stats(50)
        logging.info('Profiler: \n%s', pformat(results.getvalue()))
    ## ========================              PROFILER              ======================== ##


if __name__ == '__main__':
    _main()
