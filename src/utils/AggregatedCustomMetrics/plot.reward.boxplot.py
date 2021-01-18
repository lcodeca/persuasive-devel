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

EXPERIMENTS_BY_REWARD_MODEL = {
    'wSimplifiedReward_noBGTraffic': [
        'ppo_1000ag_5m_wParetoDistr_30_2_30_3_45_4_45_4_60_5_60_5_60_5_60_5_wGlobalUsage_wFutureDemand_wSimplifiedReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'ppo_1000ag_5m_wParetoDistr_30_2_wGlobalUsage_wFutureDemand_wSimplifiedReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'ppo_1000ag_5m_wParetoDistr_30_3_wGlobalUsage_wFutureDemand_wSimplifiedReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'ppo_1000ag_5m_wParetoDistr_45_4_wGlobalUsage_wFutureDemand_wSimplifiedReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimplifiedReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
    ],
    'wSimpleTTReward_noBGTraffic': [
        'ppo_1000ag_5m_wParetoDistr_30_2_30_3_45_4_45_4_60_5_60_5_60_5_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'ppo_1000ag_5m_wParetoDistr_30_2_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'ppo_1000ag_5m_wParetoDistr_30_3_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'ppo_1000ag_5m_wParetoDistr_45_4_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
    ],
    'wSimpleTTCoopReward_noBGTraffic': [
        'ppo_1000ag_5m_wParetoDistr_30_2_30_3_45_4_45_4_60_5_60_5_60_5_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'ppo_1000ag_5m_wParetoDistr_30_2_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'ppo_1000ag_5m_wParetoDistr_30_3_GlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'ppo_1000ag_5m_wParetoDistr_45_4_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
    ],
    'wSimpleTTCoopReward_wBGTraffic': [
        'ppo_1000ag_5m_wParetoDistr_30_2_30_3_45_4_45_4_60_5_60_5_60_5_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_wBGTraffic_deep100_1000_128',
    ]
}

EXPERIMENTS_BY_INITIAL_DISTRIBUTION = {
    'complete': [
        'ppo_1000ag_5m_wParetoDistr_30_2_30_3_45_4_45_4_60_5_60_5_60_5_60_5_wGlobalUsage_wFutureDemand_wSimplifiedReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'ppo_1000ag_5m_wParetoDistr_30_2_30_3_45_4_45_4_60_5_60_5_60_5_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'ppo_1000ag_5m_wParetoDistr_30_2_30_3_45_4_45_4_60_5_60_5_60_5_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'ppo_1000ag_5m_wParetoDistr_30_2_30_3_45_4_45_4_60_5_60_5_60_5_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_wBGTraffic_deep100_1000_128',
    ],
    '30_2': [
        'ppo_1000ag_5m_wParetoDistr_30_2_wGlobalUsage_wFutureDemand_wSimplifiedReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'ppo_1000ag_5m_wParetoDistr_30_2_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'ppo_1000ag_5m_wParetoDistr_30_2_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
    ],
    '30_3': [
        'ppo_1000ag_5m_wParetoDistr_30_3_wGlobalUsage_wFutureDemand_wSimplifiedReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'ppo_1000ag_5m_wParetoDistr_30_3_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'ppo_1000ag_5m_wParetoDistr_30_3_GlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
    ],
    '45_4': [
        'ppo_1000ag_5m_wParetoDistr_45_4_wGlobalUsage_wFutureDemand_wSimplifiedReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'ppo_1000ag_5m_wParetoDistr_45_4_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'ppo_1000ag_5m_wParetoDistr_45_4_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
    ],
    '60_5': [
        'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimplifiedReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
        'ppo_1000ag_5m_wParetoDistr_60_5_wGlobalUsage_wFutureDemand_wSimpleTTCoopReward_StochasticSampling_wMetrics_wEval_noBGTraffic_deep100_1000_128',
    ],

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

    Reward(config.input, config.output, config.outliers).generate()

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler.disable()
        results = io.StringIO()
        pstats.Stats(profiler, stream=results).sort_stats('cumulative').print_stats(50)
        print('Profiler: \n{}'.format(pformat(results.getvalue())))
    ## ========================              PROFILER              ======================== ##

####################################################################################################

class Reward():
    """ Boxplots for the rewards, compared by distribution, divided by reward model. """

    def __init__(self, input_data, output_file, outliers):
        self._input = input_data
        self._output = output_file
        self._outliers = outliers
        with open(input_data, 'r') as jsonfile:
            self._complete_data = json.load(jsonfile)

    def generate(self):
        fig, axs = plt.subplots(1, 4, figsize=(25, 15), constrained_layout=True)
        fig.suptitle('Agent Reward by Reward Model')

        current = []
        for exp in EXPERIMENTS_BY_REWARD_MODEL['wSimplifiedReward_noBGTraffic']:
            if exp in self._complete_data:
                current.append(self._complete_data[exp])
            else:
                current.append([])
        axs[0].axhline(y=0, linestyle=':')
        axs[0].boxplot(current, showfliers=self._outliers, showmeans=True)
        axs[0].set(ylabel='Reward [#]', title='wSimplifiedReward\nnoBGTraffic')
        axs[0].set_xticks(
            range(1, len(EXPERIMENTS_BY_INITIAL_DISTRIBUTION)+1)) #, minor=False)
        axs[0].set_xticklabels(
            EXPERIMENTS_BY_INITIAL_DISTRIBUTION.keys(), rotation=45) #, fontdict=None, minor=False)
        axs[0].grid()

        current = []
        for exp in EXPERIMENTS_BY_REWARD_MODEL['wSimpleTTReward_noBGTraffic']:
            if exp in self._complete_data:
                current.append(self._complete_data[exp])
            else:
                current.append([])
        axs[1].axhline(y=0, linestyle=':')
        axs[1].boxplot(current, showfliers=self._outliers, showmeans=True)
        axs[1].set(ylabel='Reward [#]', title='wSimpleTTReward\nnoBGTraffic')
        axs[1].set_xticks(
            range(1, len(EXPERIMENTS_BY_INITIAL_DISTRIBUTION)+1)) #, minor=False)
        axs[1].set_xticklabels(
            EXPERIMENTS_BY_INITIAL_DISTRIBUTION.keys(), rotation=45) #, fontdict=None, minor=False)
        axs[1].grid()

        current = []
        for exp in EXPERIMENTS_BY_REWARD_MODEL['wSimpleTTCoopReward_noBGTraffic']:
            if exp in self._complete_data:
                current.append(self._complete_data[exp])
            else:
                current.append([])
        axs[2].axhline(y=0, linestyle=':')
        axs[2].boxplot(current, showfliers=self._outliers, showmeans=True)
        axs[2].set(ylabel='Reward [#]', title='wSimpleTTCoopReward\nnoBGTraffic')
        axs[2].set_xticks(
            range(1, len(EXPERIMENTS_BY_INITIAL_DISTRIBUTION)+1)) #, minor=False)
        axs[2].set_xticklabels(
            EXPERIMENTS_BY_INITIAL_DISTRIBUTION.keys(), rotation=45) #, fontdict=None, minor=False)
        axs[2].grid()

        current = []
        for exp in EXPERIMENTS_BY_REWARD_MODEL['wSimpleTTCoopReward_wBGTraffic']:
            if exp in self._complete_data:
                current.append(self._complete_data[exp])
            else:
                current.append([])
        axs[3].axhline(y=0, linestyle=':')
        axs[3].boxplot(current, showfliers=self._outliers, showmeans=True)
        axs[3].set(ylabel='Reward [#]', title='wSimpleTTCoopReward\nwBGTraffic')
        axs[3].set_xticks(
            range(1, len(EXPERIMENTS_BY_INITIAL_DISTRIBUTION)+1)) #, minor=False)
        axs[3].set_xticklabels(
            EXPERIMENTS_BY_INITIAL_DISTRIBUTION.keys(), rotation=45) #, fontdict=None, minor=False)
        axs[3].grid()

        # plt.show()
        fig.savefig(self._output, dpi=300, transparent=False, bbox_inches='tight')
        matplotlib.pyplot.close('all')

####################################################################################################

if __name__ == '__main__':
    _main()

####################################################################################################

# ###############################################################################

# green_diamond = dict(markerfacecolor='g', marker='D')
# ax3.set_title('Changed Outlier Symbols')
# ax3.boxplot(data, flierprops=green_diamond)

# ###############################################################################

# fig4, ax4 = plt.subplots()
# ax4.set_title('Hide Outlier Points')
# ax4.boxplot(data, showfliers=self._outliers, showmeans=True)

# plt.show()

# ###############################################################################

# red_square = dict(markerfacecolor='r', marker='s')
# fig5, ax5 = plt.subplots()
# ax5.set_title('Horizontal Boxes')
# ax5.boxplot(data, vert=False, flierprops=red_square)

# plt.show()

# ###############################################################################

# fig6, ax6 = plt.subplots()
# ax6.set_title('Shorter Whisker Length')
# ax6.boxplot(data, flierprops=red_square, vert=False, whis=0.75)

# plt.show()

# ###############################################################################
# # Fake up some more data

# spread = np.random.rand(50) * 100
# center = np.ones(25) * 40
# flier_high = np.random.rand(10) * 100 + 100
# flier_low = np.random.rand(10) * -100
# d2 = np.concatenate((spread, center, flier_high, flier_low))
# data.shape = (-1, 1)
# d2.shape = (-1, 1)

# ###############################################################################
# # Making a 2-D array only works if all the columns are the
# # same length.  If they are not, then use a list instead.
# # This is actually more efficient because boxplot converts
# # a 2-D array into a list of vectors internally anyway.

# data = [data, d2, d2[::2,0]]
# fig7, ax7 = plt.subplots()
# ax7.set_title('Multiple Samples with Different sizes')
# ax7.boxplot(data)

# plt.show()
