#!/usr/bin/env python3

""" Process the RLLIB metrics_XYZ.json """

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

from genericgraphmaker import GenericGraphMaker

####################################################################################################

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
    parser.add_argument('--input-dir', required=True, type=str,
                        help='Input JSONs directory.')
    parser.add_argument('--output-dir', required=True, type=str,
                        help='Output aggregation & graphs directory.')
    parser.add_argument('--profiler', dest='profiler', action='store_true',
                        help='Enable cProfile.')
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

    TotalReward(config.input_dir, config.output_dir).generate()
    logging.info('Done')

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler.disable()
        results = io.StringIO()
        pstats.Stats(profiler, stream=results).sort_stats('cumulative').print_stats(50)
        logging.info('Profiler: \n%s', pformat(results.getvalue()))
    ## ========================              PROFILER              ======================== ##

####################################################################################################

class TotalReward(GenericGraphMaker):

    def __init__(self, input_dir, output_dir):
        _default = {
            'learning': {
                'timesteps_total': [],
                'episode_reward_min': [],
                'episode_reward_mean': [],
                'episode_reward_max':[],
            },
            'evaluation': {
                'timesteps_total': [],
                'episode_reward_min': [],
                'episode_reward_mean': [],
                'episode_reward_max':[],
            },
        }
        super().__init__(
            input_dir, output_dir,
            filename='total-reward.json',
            default=_default)

    def _find_last_metric(self):
        return len(self._aggregated_dataset['learning']['timesteps_total'])

    def _aggregate_metrics(self, files):
        for filename in tqdm(files):
            # print(filename)
            with open(os.path.join(self._input_dir, filename), 'r') as jsonfile:
                complete = json.load(jsonfile)

                # LEARNING
                self._aggregated_dataset['learning']['timesteps_total'].append(
                    complete['timesteps_total'])
                self._aggregated_dataset['learning']['episode_reward_min'].append(
                    complete['episode_reward_min'])
                self._aggregated_dataset['learning']['episode_reward_mean'].append(
                    complete['episode_reward_mean'])
                self._aggregated_dataset['learning']['episode_reward_max'].append(
                    complete['episode_reward_max'])

                # EVALUATION
                if 'evaluation' in complete:
                    complete['evaluation']['timesteps_total'] = complete['timesteps_total']
                    complete = complete['evaluation']

                    self._aggregated_dataset['evaluation']['timesteps_total'].append(
                        complete['timesteps_total'])
                    self._aggregated_dataset['evaluation']['episode_reward_min'].append(
                        complete['episode_reward_min'])
                    self._aggregated_dataset['evaluation']['episode_reward_mean'].append(
                        complete['episode_reward_mean'])
                    self._aggregated_dataset['evaluation']['episode_reward_max'].append(
                        complete['episode_reward_max'])

    def _generate_graphs(self):
        # LEARNING
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.plot(
            self._aggregated_dataset['learning']['timesteps_total'],
            self._aggregated_dataset['learning']['episode_reward_mean'],
            '-o', label='Mean')
        ax.plot(
            self._aggregated_dataset['learning']['timesteps_total'],
            self._aggregated_dataset['learning']['episode_reward_min'], label='Min')
        ax.plot(
            self._aggregated_dataset['learning']['timesteps_total'],
            self._aggregated_dataset['learning']['episode_reward_max'], label='Max')
        ax.set(xlabel='Learning step', ylabel='Reward [#]',
               title='[L] Reward Over Time')
        ax.legend(loc='best', ncol=4, shadow=True)
        ax.grid()
        fig.savefig('{}/learning.total_reward_over_learning.svg'.format(self._output_dir),
                    dpi=300, transparent=False, bbox_inches='tight')
        #plt.show()
        matplotlib.pyplot.close('all')

        # EVALUATION
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.plot(
            self._aggregated_dataset['evaluation']['timesteps_total'],
            self._aggregated_dataset['evaluation']['episode_reward_mean'],
            '-o', label='Mean')
        ax.plot(
            self._aggregated_dataset['evaluation']['timesteps_total'],
            self._aggregated_dataset['evaluation']['episode_reward_min'], label='Min')
        ax.plot(
            self._aggregated_dataset['evaluation']['timesteps_total'],
            self._aggregated_dataset['evaluation']['episode_reward_max'], label='Max')
        ax.set(xlabel='Learning step', ylabel='Reward [#]',
               title='[E] Reward Over Time')
        ax.legend(loc='best', ncol=4, shadow=True)
        ax.grid()
        fig.savefig('{}/evaluation.total_reward_over_learning.svg'.format(self._output_dir),
                    dpi=300, transparent=False, bbox_inches='tight')
        #plt.show()
        matplotlib.pyplot.close('all')

####################################################################################################

# def total_reward_over_timesteps_total(self):
#         logger.info('Computing the reward over the timesteps total.')
#         x_coords = []
#         y_coords = []
#         min_y = []
#         max_y = []
#         with open(self.input, 'r') as jsonfile:
#             counter = 0
#             for row in tqdm(jsonfile): # enumerate cannot be used due to the size of the file
#                 complete = json.loads(row)
#                 if self.evaluation:
#                     if 'evaluation' in complete:
#                         tmp = complete['evaluation']
#                         tmp['timesteps_total'] = complete['timesteps_total']
#                         complete = tmp
#                     else:
#                         # evaluation stats requested but not present in the results
#                         continue

#                 x_coords.append(complete['timesteps_total'])
#                 y_coords.append(complete['episode_reward_mean'])
#                 min_y.append(complete['episode_reward_min'])
#                 max_y.append(complete['episode_reward_max'])
#                 counter += 1

#         fig, ax = plt.subplots(figsize=(15, 10))
#         ax.errorbar(x_coords, y_coords, capsize=5, label='Mean', fmt='-o')
#         ax.plot(x_coords, min_y, label='Min')
#         ax.plot(x_coords, max_y, label='Max')
#         ax.set(xlabel='Learning step', ylabel='Reward', title='Reward over time')
#         ax.legend(loc='best', ncol=4, shadow=True)
#         ax.grid()
#         fig.savefig('{}.total_reward_over_learning.svg'.format(self.prefix),
#                     dpi=300, transparent=False, bbox_inches='tight')
#         #plt.show()
#         # ZOOM IT
#         # ax.set_ylim(-20000, 0)
#         # # plt.show()
#         # fig.savefig('{}.bounded_reward_over_learning.svg'.format(self.prefix),
#         #             dpi=300, transparent=False, bbox_inches='tight')
#         matplotlib.pyplot.close('all')

####################################################################################################

if __name__ == '__main__':
    _main()

####################################################################################################
