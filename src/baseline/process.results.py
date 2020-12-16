#!/usr/bin/env python3

""" Read all the simulation results from a series of folders. """

import argparse
import collections
import csv
import cProfile
import io
import json
import math
import os
import pstats
import sys
import traceback
import statistics

from datetime import datetime
from decimal import Decimal
from pprint import pprint, pformat
from tqdm import tqdm

import numpy as np
from numpy.random import RandomState
import shapely.geometry as geometry

# """ Import SUMO library """
if 'SUMO_TOOLS' in os.environ:
    sys.path.append(os.environ['SUMO_TOOLS'])
    import traci
    import sumolib
    from traci.exceptions import TraCIException
    import traci.constants as tc
else:
    sys.exit("Please declare environment variable 'SUMO_TOOLS'")

####################################################################################################

def argument_parser():
    """ Argument parser for the trainer"""
    parser = argparse.ArgumentParser(description='Process the BASELINE for Persuasive.')
    parser.add_argument(
        '--folders', nargs='+', default=[], help='List of folders containing the JSON results.')
    parser.add_argument('--output', required=True, type=str, help='Output directory.')
    parser.add_argument('--profiler', action='store_true', help='Enables cProfile.')
    parser.set_defaults(profiler=False)
    return parser.parse_args()

ARGS = argument_parser()

####################################################################################################

def _main():
    """ Example of integration of triggers with PyPML. """

    for folder in ARGS.folders:
        print('Processing: {}'.format(folder))

        # METRICS
        arrivals = []
        departures = []
        latenesses = []
        num_lates = []
        modes = collections.defaultdict(list)
        travel_times = []
        waitings = []

        for item in tqdm(os.listdir(folder)):
            if '.json' not in item:
                continue
            res = {
                'arrival': [],
                'departure': [],
                'lateness': [],
                'num_late': 0,
                'mode': collections.defaultdict(int),
                'travel': [],
                'waiting': [],
            }
            with open(os.path.join(folder, item), 'r') as resfile:
                simulation_results = json.load(resfile)
                for agent, results in simulation_results.items():
                    if agent == 'global':
                        continue
                    for metric, val in results.items():
                        if metric == 'mode':
                            res['mode'][val] += 1
                        elif metric == 'lateness' and not np.isnan(val):
                            res['num_late'] += 1
                            res[metric].append(val)
                        elif metric in res:
                            res[metric].append(val)
            arrivals.append(np.nanmean(res['arrival']))
            departures.append(np.nanmean(res['departure']))
            latenesses.append(np.nanmean(res['lateness']))
            num_lates.append(res['num_late'])
            travel_times.append(np.nanmean(res['travel']))
            waitings.append(np.nanmean(res['waiting']))
            for mode, val in res['mode'].items():
                modes[mode].append(val)

        # All gathered in the folder
        report = {}
        report['id'] = folder
        report['departure_mean'] = np.nanmean(departures)
        report['departure_median'] = np.nanmedian(departures)
        report['departure_std'] = np.nanstd(departures)
        report['departure_min'] = np.nanmin(departures)
        report['departure_max'] = np.nanmax(departures)
        report['arrivals_mean'] = np.nanmean(arrivals)
        report['arrivals_median'] = np.nanmedian(arrivals)
        report['arrivals_std'] = np.nanstd(arrivals)
        report['arrivals_min'] = np.nanmin(arrivals)
        report['arrivals_max'] = np.nanmax(arrivals)
        report['travel_times_mean'] = np.nanmean(travel_times)
        report['travel_times_median'] = np.nanmedian(travel_times)
        report['travel_times_std'] = np.nanstd(travel_times)
        report['travel_times_min'] = np.nanmin(travel_times)
        report['travel_times_max'] = np.nanmax(travel_times)
        report['waitings_mean'] = np.nanmean(waitings)
        report['waitings_median'] = np.nanmedian(waitings)
        report['waitings_std'] = np.nanstd(waitings)
        report['waitings_min'] = np.nanmin(waitings)
        report['waitings_max'] = np.nanmax(waitings)
        report['latenesses_mean'] = np.nanmean(latenesses)
        report['latenesses_median'] = np.nanmedian(latenesses)
        report['latenesses_std'] = np.nanstd(latenesses)
        report['latenesses_min'] = np.nanmin(latenesses)
        report['latenesses_max'] = np.nanmax(latenesses)
        report['num_lates_mean'] = np.nanmean(num_lates)
        report['num_lates_median'] = np.nanmedian(num_lates)
        report['num_lates_std'] = np.nanstd(num_lates)
        report['num_lates_min'] = np.nanmin(num_lates)
        report['num_lates_max'] = np.nanmax(num_lates)
        for mode, values in modes.items():
            report[mode] = np.nanmean(values)
        pprint(report)
        if os.path.isfile(ARGS.output):
            with open(ARGS.output, 'a') as f:
                w = csv.DictWriter(f, sorted(report.keys()))
                w.writerow(report)
        else: # Create with the header
            with open(ARGS.output, 'w') as f:
                w = csv.DictWriter(f, sorted(report.keys()))
                w.writeheader()
                w.writerow(report)
    print('Saved:', ARGS.output)

if __name__ == '__main__':
    ## ========================              PROFILER              ======================== ##
    if ARGS.profiler:
        profiler = cProfile.Profile()
        profiler.enable()
    ## ========================              PROFILER              ======================== ##
    _main()
    ## ========================              PROFILER              ======================== ##
    if ARGS.profiler:
        profiler.disable()
        output = io.StringIO()
        pstats.Stats(profiler, stream=output).sort_stats('cumulative').print_stats(25)
        print(output.getvalue())
    ## ========================              PROFILER              ======================== ##

####################################################################################################
