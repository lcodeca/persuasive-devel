#!/usr/bin/env python3

""" Process the DBLogger directory structure generating Q-values plots for specific episodes. """

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
from math import ceil
from deepdiff import DeepDiff
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from dbloggerstats import DBLoggerStats

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

TINY_SIZE = 10
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
    """ Process the DBLogger directory structure generating Q-values plots for specific episodes. """

    config = _argument_parser()

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler = cProfile.Profile()
        profiler.enable()
    ## ========================              PROFILER              ======================== ##

    statistics = QValuesInsight(
        config.dir_tree, config.graph, config.training, config.agent, config.last_run)
    statistics.generate_plots()
    LOGGER.info('Done')

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler.disable()
        results = io.StringIO()
        pstats.Stats(profiler, stream=results).sort_stats('cumulative').print_stats(50)
        LOGGER.info('Profiler: \n%s', pformat(results.getvalue()))
    ## ========================              PROFILER              ======================== ##

class QValuesInsight(DBLoggerStats):
    """ 
        Process the DBLogger directory structure generating Q-values plots for specific episodes. 
    """

    def __init__(self, directory, prefix, training, agent, last):
        super().__init__(directory)
        self.output_prefix = prefix
        self.training = training
        self.agent = agent
        self.last = last
        if self.last:
            self.training, self.agent = None, None

    ######################################## PLOT GENERATOR ########################################

    def generate_plots(self):
        available_training_runs = list()
        if self.training:
            available_training_runs.append(self.training)
        else:
            available_training_runs = self.alphanumeric_sort(os.listdir(self.dir))
        
        if self.last:
            available_training_runs = [available_training_runs[-1]]

        print('Processing the training runs...')
        for training_run in tqdm(available_training_runs):
            available_agents, _, _ = self.get_training_components(training_run)
            if self.agent:
                available_agents = [self.agent]
            print('Processing agents...')
            for agent in tqdm(available_agents):
                qtable = self.get_qtable(training_run, agent)
                state_action_counter = self.get_state_action_counter(training_run, agent)
                structure = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: list())))
                for state, counters in state_action_counter.items():
                    for action, counter in counters.items():
                        # if counter <= 0:
                        #     # irrelevant
                        #     continue
                        time_left_match = re.search('\(\'time-left\', (.+?)\), \(\'ett\'', state)
                        if not time_left_match:
                            # this should never happen but..
                            continue
                        time_left = time_left_match.group(1)
                        ett_match = re.search('array\((.+?)\)\)]\)', state)
                        if not ett_match:
                            # this should never happen but..
                            continue
                        ett = ett_match.group(1)
                        structure[int(time_left)][ett][int(action)] = [round(qtable[state][str(action)], 1), counter]

                labels = [] # ordered states, time left - ETTs
                dict_qvals = defaultdict(list)
                dict_counters = defaultdict(list)
                for time_left, values in sorted(structure.items()):
                    for ett, actions in sorted(values.items()):
                        labels.append('{}-{}'.format(time_left, ett))
                        for action, (qval, counter) in actions.items():
                            dict_qvals[action].append(qval)
                            dict_counters[action].append(counter)
                
                ## PLOTTING TIME!
                x = np.arange(len(labels))  # the label locations
                width = 0.15  # the width of the bars

                fig, ax = plt.subplots(figsize=(70, 30))
                spacer = len(dict_qvals)
                for pos, action in enumerate(dict_qvals):
                    relpos = ceil(pos - spacer / 2) / spacer / 2
                    # print(action, relpos, pos, spacer)
                    # print(x + relpos * width / 2)
                    rect = ax.bar(x + relpos + width / 2, dict_qvals[action], width, label='{}'.format(action))
                    for p, bar in enumerate(rect):
                        if dict_counters[action][p] > 0:
                            lbl = '({}) {}'.format(dict_counters[action][p], dict_qvals[action][p])
                            ax.annotate(lbl, # dict_counters[action][p],
                                        xy=(bar.get_x() + bar.get_width() / spacer, bar.get_height()),
                                        xytext=(0, -10),  # 3 points vertical offset
                                        textcoords="offset points", size=TINY_SIZE, 
                                        rotation=90, ha='center', va='top')

                # Add some text for labels, title and custom x-axis tick labels, etc.
                ax.set_ylabel('Q-Values')
                ax.set_title('Q-values by state and action')
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation='vertical')
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position('top') 
                ax.legend()
                
                fig.savefig('{}.{}.{}.svg'.format(
                                self.output_prefix, training_run, agent),
                            dpi=300, transparent=False, bbox_inches='tight')
                # fig.savefig('{}.{}.{}.png'.format(
                #                 self.output_prefix, training_run, agent),
                #             dpi=300, transparent=False, bbox_inches='tight')
                # plt.show()   
                matplotlib.pyplot.close('all')

####################################################################################################

if __name__ == '__main__':
    _main()

####################################################################################################