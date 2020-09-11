#!/usr/bin/env python3

""" Process the DBLogger directory structure generating Policiy plots for specific episodes. """

import argparse
from collections import defaultdict
import cProfile
import io
import logging
import os
from pprint import pformat
import pstats
import re

import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from dbloggerstats import DBLoggerStats

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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
    """ Process the DBLogger directory structure generating Policiy plots for specific episodes. """

    config = _argument_parser()

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler = cProfile.Profile()
        profiler.enable()
    ## ========================              PROFILER              ======================== ##

    statistics = PolicyInsight(
        config.dir_tree, config.graph, config.training, config.agent, config.last_run)
    statistics.generate_plots()
    logger.info('Done')

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler.disable()
        results = io.StringIO()
        pstats.Stats(profiler, stream=results).sort_stats('cumulative').print_stats(50)
        logger.info('Profiler: \n%s', pformat(results.getvalue()))
    ## ========================              PROFILER              ======================== ##

class PolicyInsight(DBLoggerStats):

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
                best_action = self.get_best_actions(training_run, agent)
                state_action_counter = self.get_state_action_counter(training_run, agent)
                structure = defaultdict(lambda: defaultdict(lambda: list()))
                for state, action in best_action.items():
                    time_left_match = re.search('\(\'time-left\', (.+?)\), \(\'ett\'', state)
                    time_left = None
                    if time_left_match:
                        time_left = time_left_match.group(1)
                    ett_match = re.search('array\((.+?)\)\)]\)', state)
                    ett = None
                    if ett_match:
                        ett = ett_match.group(1)
                    if time_left is None or ett is None:
                        print('Problem with state ', state)
                    else:
                        counter = 0
                        try:
                            counter = state_action_counter[state][str(action)]
                        except KeyError:
                            pass
                        if counter > 0:
                            structure[int(time_left)][int(action)].append(
                                '{} {}'.format(counter, ett))

                x_coord = []
                y_coord = []
                labels = []
                for time_left, values in structure.items():
                    for action, all_labels in values.items():
                        aggr = ''
                        for lbl in all_labels:
                            aggr += ' {} /'.format(lbl)
                        aggr = aggr.strip('/')
                        x_coord.append(time_left)
                        y_coord.append(action)
                        labels.append(aggr)

                ## PLOTTING TIME!
                fig, ax = plt.subplots(figsize=(30, 10))
                ax.scatter(x_coord, y_coord, label='Best Action')
                ax.set_ylim(-0.5, max(y_coord, default=0.0)+0.5)
                ax.set_yticks(range(0, max(y_coord, default=0)+1, 1))
                for i, txt in enumerate(labels):
                    ax.annotate(txt, (x_coord[i], y_coord[i]), rotation=90,
                                horizontalalignment='center')
                ax.set(xlabel='Waiting slots', ylabel='Action [#]',
                       title='Best Action "{}" during {}.'.format(agent, training_run))
                ax.grid()
                fig.savefig('{}.{}.{}.svg'.format(self.output_prefix, training_run, agent),
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
