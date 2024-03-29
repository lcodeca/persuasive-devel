#!/usr/bin/env python3

""" Plot the aggregated total rewards. """

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

    TotReward(config.input, config.output, config.outliers).generate()

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler.disable()
        results = io.StringIO()
        pstats.Stats(profiler, stream=results).sort_stats('cumulative').print_stats(50)
        print('Profiler: \n{}'.format(pformat(results.getvalue())))
    ## ========================              PROFILER              ======================== ##

####################################################################################################

class TotReward():
    """ Boxplots for the rewards, compared by distribution, divided by reward model. """

    def __init__(self, input_data, output_file, outliers):
        self._input = input_data
        self._output = output_file
        self._outliers = outliers
        with open(input_data, 'r') as jsonfile:
            self._complete_data = json.load(jsonfile)

    def _nan_to_0(self, array):
        ret = []
        for val in array:
            if not np.isnan(val):
                ret.append(val)
        if ret:
            return ret
        return [0]

    def _pack_data(self, chunk):
        current = []
        labels = []
        for lbl, exp in consts.EXPERIMENTS[chunk].items():
            labels.append(lbl)
            if exp in self._complete_data:
                current.append(self._nan_to_0(self._complete_data[exp]))
            else:
                current.append([])
        return labels, current

    def generate(self):
        fig, axs = plt.subplots(1, 5, figsize=(25, 15), constrained_layout=True)
        fig.suptitle('Episode Total Reward by Initial Distribution')

        labels, current = self._pack_data('All')
        axs[0].axhline(y=0, linestyle=':')
        axs[0].boxplot(current, showfliers=self._outliers, showmeans=True)
        axs[0].set(ylabel='Reward [#]', title='All')
        axs[0].set_xticks(range(1, len(labels)+1)) #, minor=False)
        axs[0].set_xticklabels(labels, rotation=90) #, fontdict=None, minor=False)
        axs[0].grid()

        labels, current = self._pack_data('30_2')
        axs[1].axhline(y=0, linestyle=':')
        axs[1].boxplot(current, showfliers=self._outliers, showmeans=True)
        axs[1].set(ylabel='Reward [#]', title='30_2')
        axs[1].set_xticks(range(1, len(labels)+1)) #, minor=False)
        axs[1].set_xticklabels(labels, rotation=90) #, fontdict=None, minor=False)
        axs[1].grid()

        labels, current = self._pack_data('30_3')
        axs[2].axhline(y=0, linestyle=':')
        axs[2].boxplot(current, showfliers=self._outliers, showmeans=True)
        axs[2].set(ylabel='Reward [#]', title='30_3')
        axs[2].set_xticks(range(1, len(labels)+1)) #, minor=False)
        axs[2].set_xticklabels(labels, rotation=90) #, fontdict=None, minor=False)
        axs[2].grid()

        labels, current = self._pack_data('45_4')
        axs[3].axhline(y=0, linestyle=':')
        axs[3].boxplot(current, showfliers=self._outliers, showmeans=True)
        axs[3].set(ylabel='Reward [#]', title='45_4')
        axs[3].set_xticks(range(1, len(labels)+1)) #, minor=False)
        axs[3].set_xticklabels(labels, rotation=90) #, fontdict=None, minor=False)
        axs[3].grid()

        labels, current = self._pack_data('60_5')
        axs[4].axhline(y=0, linestyle=':')
        axs[4].boxplot(current, showfliers=self._outliers, showmeans=True)
        axs[4].set(ylabel='Reward [#]', title='60_5')
        axs[4].set_xticks(range(1, len(labels)+1)) #, minor=False)
        axs[4].set_xticklabels(labels, rotation=90) #, fontdict=None, minor=False)
        axs[4].grid()

        # plt.show()
        fig.savefig(self._output, dpi=300, transparent=False, bbox_inches='tight')
        matplotlib.pyplot.close('all')

    # def generate(self):
    #     fig, axs = plt.subplots(1, 4, figsize=(25, 15), constrained_layout=True)
    #     fig.suptitle('Episode Total Reward by Reward Model')

    #     labels, current = self._pack_data('wSimplifiedReward_noBGTraffic')
    #     axs[0].axhline(y=0, linestyle=':')
    #     axs[0].boxplot(current, showfliers=self._outliers, showmeans=True)
    #     axs[0].set(ylabel='Reward [#]', title='wSimplifiedReward\nnoBGTraffic')
    #     axs[0].set_xticks(range(1, len(labels)+1)) #, minor=False)
    #     axs[0].set_xticklabels(labels, rotation=90) #, fontdict=None, minor=False)
    #     axs[0].grid()

    #     labels, current = self._pack_data('wSimpleTTReward_noBGTraffic')
    #     axs[1].axhline(y=0, linestyle=':')
    #     axs[1].boxplot(current, showfliers=self._outliers, showmeans=True)
    #     axs[1].set(ylabel='Reward [#]', title='wSimpleTTReward\nnoBGTraffic')
    #     axs[1].set_xticks(range(1, len(labels)+1)) #, minor=False)
    #     axs[1].set_xticklabels(labels, rotation=90) #, fontdict=None, minor=False)
    #     axs[1].grid()

    #     labels, current = self._pack_data('wSimpleTTCoopReward_noBGTraffic')
    #     axs[2].axhline(y=0, linestyle=':')
    #     axs[2].boxplot(current, showfliers=self._outliers, showmeans=True)
    #     axs[2].set(ylabel='Reward [#]', title='wSimpleTTCoopReward\nnoBGTraffic')
    #     axs[2].set_xticks(range(1, len(labels)+1)) #, minor=False)
    #     axs[2].set_xticklabels(labels, rotation=90) #, fontdict=None, minor=False)
    #     axs[2].grid()

    #     labels, current = self._pack_data('wSimpleTTCoopReward_wBGTraffic')
    #     axs[3].axhline(y=0, linestyle=':')
    #     axs[3].boxplot(current, showfliers=self._outliers, showmeans=True)
    #     axs[3].set(ylabel='Reward [#]', title='wSimpleTTCoopReward\nwBGTraffic')
    #     axs[3].set_xticks(range(1, len(labels)+1)) #, minor=False)
    #     axs[3].set_xticklabels(labels, rotation=90) #, fontdict=None, minor=False)
    #     axs[3].grid()

    #     # plt.show()
    #     fig.savefig(self._output, dpi=300, transparent=False, bbox_inches='tight')
    #     matplotlib.pyplot.close('all')

####################################################################################################

if __name__ == '__main__':
    _main()

####################################################################################################
