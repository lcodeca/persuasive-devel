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
    'ppo_1000ag_5m_wParetoDistr_30_2_30_3_45_4_45_4_60_5_60_5_60_5_60_5_wGlobalUsage_wFutureDemand_wSimplifiedReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128': {
        'title': 'wSimplifiedReward\nnoBGTraffic\nComplete',
        'coords': (0, 0),
    },
    'ppo_1000ag_5m_wParetoDistr_30_2_wGlobalUsage_wFutureDemand_wSimplifiedReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128': {
        'title': 'wSimplifiedReward\nnoBGTraffic\n30_2',
        'coords': (1, 0),
    },
    'ppo_1000ag_5m_wParetoDistr_30_3_wGlobalUsage_wFutureDemand_wSimplifiedReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128': {
        'title': 'wSimplifiedReward\nnoBGTraffic\n30_3',
        'coords': (2, 0),
    },
    'ppo_1000ag_5m_wParetoDistr_45_4_wGlobalUsage_wFutureDemand_wSimplifiedReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128': {
        'title': 'wSimplifiedReward\nnoBGTraffic\n45_4',
        'coords': (3, 0),
    },
    'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimplifiedReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128': {
        'title': 'wSimplifiedReward\nnoBGTraffic\n60_5',
        'coords': (4, 0),
    },
    'ppo_1000ag_5m_wParetoDistr_30_2_30_3_45_4_45_4_60_5_60_5_60_5_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128': {
        'title': 'wSimpleTTReward\nnoBGTraffic\nComplete',
        'coords': (0, 1),
    },
    'ppo_1000ag_5m_wParetoDistr_30_2_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128': {
        'title': 'wSimpleTTReward\nnoBGTraffic\n30_2',
        'coords': (1, 1),
    },
    'ppo_1000ag_5m_wParetoDistr_30_3_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128': {
        'title': 'wSimpleTTReward\nnoBGTraffic\n30_3',
        'coords': (2, 1),
    },
    'ppo_1000ag_5m_wParetoDistr_45_4_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128': {
        'title': 'wSimpleTTReward\nnoBGTraffic\n45_4',
        'coords': (3, 1),
    },
    'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128': {
        'title': 'wSimpleTTReward\nnoBGTraffic\n60_5',
        'coords': (4, 1),
    },
    'ppo_1000ag_5m_wParetoDistr_30_2_30_3_45_4_45_4_60_5_60_5_60_5_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128': {
        'title': 'wSimpleTTCoopReward\nnoBGTraffic\nComplete',
        'coords': (0, 2),
    },
    'ppo_1000ag_5m_wParetoDistr_30_2_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128': {
        'title': 'wSimpleTTCoopReward\nnoBGTraffic\n30_2',
        'coords': (1, 2),
    },
    'ppo_1000ag_5m_wParetoDistr_30_3_GlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128': {
        'title': 'wSimpleTTCoopReward\nnoBGTraffic\n30_3',
        'coords': (2, 2),
    },
    'ppo_1000ag_5m_wParetoDistr_45_4_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128': {
        'title': 'wSimpleTTCoopReward\nnoBGTraffic\n45_4',
        'coords': (3, 2),
    },
    'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128': {
        'title': 'wSimpleTTCoopReward\nnoBGTraffic\n60_5',
        'coords': (4, 2),
    },
    'ppo_1000ag_5m_wParetoDistr_30_2_30_3_45_4_45_4_60_5_60_5_60_5_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_wBGTraffic_deep100_1000_128': {
        'title': 'wSimpleTTCoopReward\nwBGTraffic\nComplete',
        'coords': (0, 3),
    },
    'ppo_1000ag_4m_wParetoDistr_30_2_30_3_45_4_45_4_60_5_60_5_60_5_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_noPTW': {
        'title': 'wSimpleTTReward\nnoBGTraffic\nComplete_noPTW',
        'coords': (0, 4),
    },
    'ppo_1000ag_4m_wParetoDistr_30_2_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_noPTW': {
        'title': 'wSimpleTTReward\nnoBGTraffic\n30_2_noPTW',
        'coords': (1, 4),
    },
    'ppo_1000ag_4m_wParetoDistr_30_3_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_noPTW': {
        'title': 'wSimpleTTReward\nnoBGTraffic\n30_3_noPTW',
        'coords': (2, 4),
    },
    'ppo_1000ag_4m_wParetoDistr_45_4_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_noPTW': {
        'title': 'wSimpleTTReward\nnoBGTraffic\n45_5_noPTW',
        'coords': (3, 4),
    },
    'ppo_1000ag_4m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_noPTW': {
        'title': 'wSimpleTTReward\nnoBGTraffic\n60_6_noPTW',
        'coords': (4, 4),
    },
    'ppo_1000ag_4m_wParetoDistr_30_2_30_3_45_4_45_4_60_5_60_5_60_5_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_noPTW': {
        'title': 'wSimpleTTCoopReward\nnoBGTraffic\nComplete_noPTW',
        'coords': (0, 5),
    },
    'ppo_1000ag_4m_wParetoDistr_30_2_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_noPTW': {
        'title': 'wSimpleTTCoopReward\nnoBGTraffic\n30_2_noPTW',
        'coords': (1, 5),
    },
    'ppo_1000ag_4m_wParetoDistr_30_3_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_noPTW': {
        'title': 'wSimpleTTCoopReward\nnoBGTraffic\n30_3_noPTW',
        'coords': (2, 5),
    },
    'ppo_1000ag_4m_wParetoDistr_45_4_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_noPTW': {
        'title': 'wSimpleTTCoopReward\nnoBGTraffic\n45_5_noPTW',
        'coords': (3, 5),
    },
    'ppo_1000ag_4m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_noPTW': {
        'title': 'wSimpleTTCoopReward\nnoBGTraffic\n60_6_noPTW',
        'coords': (4, 5),
    },
    'ppo_1000ag_2m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_carOnly': {
        'title': 'wSimpleTTReward\nnoBGTraffic\n60_5_carOnly',
        'coords': (4, 6),
    },
    'ppo_1000ag_2m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_carOnly': {
        'title': 'wSimpleTTCoopReward\nnoBGTraffic\n60_5_carOnly',
        'coords': (4, 7),
    },
    'ppo_1000ag_5m_wParetoDistr_30_2_30_2_45_4_45_4_60_5_60_5_60_5_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_ext': {
        'title': 'wSimpleTTReward\nnoBGTraffic\nComplete_ext',
        'coords': (0, 8),
    },
    'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_ext': {
        'title': 'wSimpleTTReward\nnoBGTraffic\n60_5_ext',
        'coords': (4, 8),
    },
    'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_metroOnly': {
        'title': 'wSimpleTTReward\nnoBGTraffic\n60_5_metroOnly',
        'coords': (4, 9),
    },
    'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_metroOnly': {
        'title': 'wSimpleTTCoopReward\nnoBGTraffic\n60_5_metroOnly',
        'coords': (4, 10),
    },
    'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_wParking': {
        'title': 'wSimpleTTReward\nnoBGTraffic\n60_5_wParking',
        'coords': (4, 11),
    },
    'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128_wParking': {
        'title': 'wSimpleTTCoopReward\nnoBGTraffic\n60_5_wParking',
        'coords': (4, 12),
    },
    'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_wOwnership_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128': {
        'title': 'wSimpleTTReward\nnoBGTraffic\n60_5_wOwnership',
        'coords': (4, 13),
    },
    'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_wOwnership_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128': {
        'title': 'wSimpleTTCoopReward\nnoBGTraffic\n60_5_wOwnership',
        'coords': (4, 14),
    },
    'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_wPreferences_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128': {
        'title': 'wSimpleTTReward\nnoBGTraffic\n60_5_wPreferences',
        'coords': (4, 15),
    },
    'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_wPreferences_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128': {
        'title': 'wSimpleTTCoopReward\nnoBGTraffic\n60_5_wPreferences',
        'coords': (4, 16),
    },
}

MISSING = [
    (1, 3), (2, 3), (3, 3), (4, 3),
    (0, 6), (1, 6), (2, 6), (3, 6),
    (0, 7), (1, 7), (2, 7), (3, 7),
    (1, 8), (2, 8), (3, 8),
    (0, 9), (1, 9), (2, 9), (3, 9),
    (0, 10), (1, 10), (2, 10), (3, 10),
    (0, 11), (1, 11), (2, 11), (3, 11),
    (0, 12), (1, 12), (2, 12), (3, 12),
    (0, 13), (1, 13), (2, 13), (3, 13),
    (0, 14), (1, 14), (2, 14), (3, 14),
    (0, 15), (1, 15), (2, 15), (3, 15),
    (0, 16), (1, 16), (2, 16), (3, 16),
]

MODES = ['wait', 'walk', 'bicycle', 'public', 'ptw', 'car']

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

    ModeUsage(config.input, config.output, config.outliers).generate()

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler.disable()
        results = io.StringIO()
        pstats.Stats(profiler, stream=results).sort_stats('cumulative').print_stats(50)
        print('Profiler: \n{}'.format(pformat(results.getvalue())))
    ## ========================              PROFILER              ======================== ##

####################################################################################################

class ModeUsage():
    """ Boxplots for the rewards, compared by distribution, divided by reward model. """

    def __init__(self, input_data, output_file, outliers):
        self._input = input_data
        self._output = output_file
        self._outliers = outliers
        self._complete_data = collections.defaultdict(dict)
        self._import_data()

    def _import_data(self):
        for mode in MODES:
            fname = 'm_{}.json'.format(mode)
            with open(os.path.join(self._input, fname), 'r') as jsonfile:
                data = json.load(jsonfile)
                for exp, values in data.items():
                    # print(values, exp, mode)
                    self._complete_data[exp][mode] = self._nan_to_0(values)

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
            5, 17, figsize=(50, 30),
            sharey=True, squeeze=True, constrained_layout=True, )
        fig.suptitle('Transporation Modes Usage')

        for exp, options in EXPERIMENTS_GRID.items():
            title = options['title']
            row, col = options['coords']
            current = []
            for mode in MODES:
                if mode in self._complete_data[exp]:
                    current.append(self._complete_data[exp][mode])
                else:
                    current.append([])
                    print('Missing', mode, exp)

            axs[row, col].axhline(y=0, linestyle=':')
            axs[row, col].axhline(y=200, linestyle=':')
            axs[row, col].boxplot(current, showfliers=self._outliers, showmeans=True)
            axs[row, col].set(title=title)
            axs[row, col].set_xticks(range(1, len(MODES)+1))
            axs[row, col].set_xticklabels(MODES, rotation=45)
            axs[row, col].grid()

        for row, col in MISSING:
            axs[row, col].axhline(y=0, linestyle=':')
            axs[row, col].axhline(y=200, linestyle=':')
            axs[row, col].set_xticks(range(1, len(MODES)+1))
            axs[row, col].set_xticklabels(MODES, rotation=45)
            axs[row, col].grid()

        # plt.show()
        fig.savefig(self._output, dpi=300, transparent=False, bbox_inches='tight',)
        matplotlib.pyplot.close('all')

####################################################################################################

if __name__ == '__main__':
    _main()

####################################################################################################
