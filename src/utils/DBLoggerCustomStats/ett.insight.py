#!/usr/bin/env python3

""" Process the DBLogger directory structure generating ETT plots for specific episodes. """

import argparse
from collections import defaultdict
import cProfile
import io
import json
import logging
import os
from pprint import pformat, pprint
import pstats
import re
import sys

from deepdiff import DeepDiff
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from dbloggerstats import DBLoggerStats

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

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

def _argument_parser():
    """ Argument parser for the stats parser. """
    parser = argparse.ArgumentParser(
        description='RLLIB & SUMO Statistics parser.')
    parser.add_argument(
        '--dir-tree', required=True, type=str, 
        help='DBLogger directory.')
    parser.add_argument(
        '--training', default=None, type=str, 
        help='Training run to parse, if not defined, it process them all. (Ignored if --last-run)')
    parser.add_argument(
        '--episode', default=None, type=str, 
        help='Episode to parse, if not defined, it process them all. (Ignored if --last-run)')
    parser.add_argument(
        '--agent', default=None, type=str, 
        help='Agent to parse, if not defined, it process them all. (Ignored if --last-run)')
    parser.add_argument(
        '--graph', required=True, 
        help='Output prefix for the graph(s).')
    parser.add_argument(
        '--last-run', dest='last_run', action='store_true', 
        help='Process all episodes and agents in the last training run.')
    parser.set_defaults(last_run=False)
    parser.add_argument(
        '--profiler', dest='profiler', action='store_true', help='Enable cProfile.')
    parser.set_defaults(profiler=False)
    return parser.parse_args()

def _main():
    """ Process the DBLogger directory structure generating ETT plots for specific episodes. """

    config = _argument_parser()

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler = cProfile.Profile()
        profiler.enable()
    ## ========================              PROFILER              ======================== ##

    statistics = ETTInsight(
        config.dir_tree, config.graph, 
        config.training, config.episode, config.agent, config.last_run)
    statistics.generate_plots()
    LOGGER.info('Done')

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler.disable()
        results = io.StringIO()
        pstats.Stats(profiler, stream=results).sort_stats('cumulative').print_stats(50)
        LOGGER.info('Profiler: \n%s', pformat(results.getvalue()))
    ## ========================              PROFILER              ======================== ##

class ETTInsight(DBLoggerStats):

    def __init__(self, directory, prefix, training, episode, agent, last):
        super().__init__(directory)
        self.output_prefix = prefix
        self.training = training
        self.episode = episode
        self.agent = agent
        self.last = last
        if self.last:
            self.training, self.episode, self.agent = None, None, None

    ######################################## PLOT GENERATOR ########################################

    def generate_plots(self):
        available_training_runs = list()
        if self.training:
            available_training_runs.append(self.training)
        else:
            available_training_runs = self.alphanumeric_sort(os.listdir(self.dir))
        
        if self.last:
            available_training_runs = [available_training_runs[-1]]

        for training_run in available_training_runs:
            available_agents, available_episodes, _ = self.get_training_components(training_run)
            if self.episode:
                available_episodes = [self.episode]
            if self.agent:
                available_agents = [self.agent]
            for episode in available_episodes:
                for agent in available_agents:
                    info = self.get_info(training_run, episode, agent)
                    if 'ext' not in info:
                        print('No ETT available for {} {} {}'.format(training_run, episode, agent))
                        continue
                    for mode, values in info['ext'].items():
                        x_coords = list(range(len(values)))
                        fig, ax = plt.subplots(figsize=(15, 10))
                        ax.plot(x_coords, values, label='ETT')
                        ax.set(xlabel='Waiting slots', ylabel='Time [s]', 
                               title='ETT variation for mode "{}" during {}/{}.'.format(
                                   mode, training_run, episode))
                        ax.grid()
                        fig.savefig('{}.{}.{}.{}.{}.svg'.format(
                                        self.output_prefix, training_run, episode, agent, mode),
                                    dpi=300, transparent=False, bbox_inches='tight')
                        fig.savefig('{}.{}.{}.{}.{}.svg'.format(
                                        self.output_prefix, training_run, episode, agent, mode),
                                    dpi=600, transparent=False, bbox_inches='tight')
                        # plt.show()   
                        matplotlib.pyplot.close('all')

####################################################################################################

if __name__ == '__main__':
    _main()

####################################################################################################