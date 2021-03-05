#!/usr/bin/env python3

""" Plot the aggregated rewards. """

import argparse
import collections
import cProfile
import io
import json
import logging
import os
import pstats
import re
import sys

from pprint import pformat, pprint
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

####################################################################################################

import consts

####################################################################################################

SMALL_SIZE = 20
MEDIUM_SIZE = SMALL_SIZE + 4
BIGGER_SIZE = MEDIUM_SIZE + 4

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

####################################################################################################

EXPERIMENTS_GRID = {
    'test': {
        'choice': {
            'title': 'wSimpleTTReward\nnoBGTraffic\nChoice',
            'ylabel': 'Agents [#]',
            'coords': (0, 0),
        },
        'late': {
            'title': 'wSimpleTTReward\nnoBGTraffic\nLate',
            'ylabel': 'Agents [#]',
            'coords': (1, 0),
        },
        'lateness': {
            'title': 'wSimpleTTReward\nnoBGTraffic\nLateness',
            'ylabel': 'Time [min]',
            'coords': (2, 0),
        },
        'waiting': {
            'title': 'wSimpleTTReward\nnoBGTraffic\nWaiting',
            'ylabel': 'Time [min]',
            'coords': (3, 0),
        },
    }
}

MISSING = [(0, 1), (1, 1), (2, 1), (3, 1)]

CHOICES = ['1st', '2nd', '3rd', 'rest']
FILES = {
    'choice': 'choice',
    'late': 'num_late',
    'lateness': 'lateness',
    'waiting': 'waiting',
}

####################################################################################################

def _argument_parser():
    """ Argument parser for the stats parser. """
    parser = argparse.ArgumentParser(
        description='RLLIB & SUMO Statistics parser.')
    parser.add_argument('--input', required=True, type=str, help='Input JSONs file.')
    parser.add_argument('--output', required=True, type=str, help='Output graph file.')
    parser.add_argument('--outliers', dest='outliers', action='store_true',
                        help='Enable cProfile.')
    parser.set_defaults(outliers=False)
    parser.add_argument('--profiler', dest='profiler', action='store_true',
                        help='Enable cProfile.')
    parser.set_defaults(profiler=False)
    return parser.parse_args()

def _main():
    """ Process the RLLIB logs/result.json """

    config = _argument_parser()
    print(config)

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler = cProfile.Profile()
        profiler.enable()
    ## ========================              PROFILER              ======================== ##

    Achievements(config.input, config.output, config.outliers).generate()

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler.disable()
        results = io.StringIO()
        pstats.Stats(profiler, stream=results).sort_stats('cumulative').print_stats(50)
        print('Profiler: \n{}'.format(pformat(results.getvalue())))
    ## ========================              PROFILER              ======================== ##

####################################################################################################

class Achievements():
    """ Boxplots for the rewards, compared by distribution, divided by reward model. """

    def __init__(self, input_data, output_file, outliers):
        self._input = input_data
        self._output = output_file
        self._outliers = outliers
        self._complete_data = collections.defaultdict(lambda: collections.defaultdict(dict))
        self._import_data()

    def _import_data(self):
        for choice in CHOICES:
            for tag, name in FILES.items():
                fname = 'ach_{}_{}.json'.format(choice, name)
                with open(os.path.join(self._input, fname), 'r') as jsonfile:
                    data = json.load(jsonfile)
                    for exp, values in data.items():
                        self._complete_data[exp][tag][choice] = self._nan_to_0(values)

    def _nan_to_0(self, array):
        ret = []
        for val in array:
            if not np.isnan(val):
                ret.append(val)
        if ret:
            return ret
        return [0]

    def generate(self):
        fig, axs = plt.subplots(
            4, 2, figsize=(50, 30), sharey='row', squeeze=True, constrained_layout=True, )
        fig.suptitle('Preference Achievements')

        for exp, plots in EXPERIMENTS_GRID.items():
            for tag, options in plots.items():
                title = options['title']
                ylabel = options['ylabel']
                row, col = options['coords']
                current = []
                for choice in CHOICES:
                    current.append(self._complete_data[exp][tag][choice])

                axs[row, col].axhline(y=0, linestyle=':')
                # axs[row, col].axhline(y=200, linestyle=':')
                axs[row, col].boxplot(current, showfliers=self._outliers, showmeans=True)
                axs[row, col].set(title=title, ylabel=ylabel)
                axs[row, col].set_xticks(range(1, len(CHOICES)+1))
                axs[row, col].set_xticklabels(CHOICES, rotation=45)
                axs[row, col].grid()

        for row, col in MISSING:
            axs[row, col].axhline(y=0, linestyle=':')
            # axs[row, col].axhline(y=200, linestyle=':')
            axs[row, col].set_xticks(range(1, len(CHOICES)+1))
            axs[row, col].set_xticklabels(CHOICES, rotation=45)
            axs[row, col].grid()

        # plt.show()
        fig.savefig(self._output, dpi=300, transparent=False, bbox_inches='tight',)
        matplotlib.pyplot.close('all')

####################################################################################################

if __name__ == '__main__':
    _main()

####################################################################################################
