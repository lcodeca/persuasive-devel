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

    Actions(config.input_dir, config.output_dir).generate()
    logging.info('Done')

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler.disable()
        results = io.StringIO()
        pstats.Stats(profiler, stream=results).sort_stats('cumulative').print_stats(50)
        logging.info('Profiler: \n%s', pformat(results.getvalue()))
    ## ========================              PROFILER              ======================== ##

####################################################################################################

class Actions(GenericGraphMaker):

    def __init__(self, input_dir, output_dir):
        _default = {
            'learning': {
                'timesteps_total': [],
                'actions_min': [],
                'actions_mean': [],
                'actions_median': [],
                'actions_std': [],
                'actions_max':[],
            },
            'evaluation': {
                'timesteps_total': [],
                'actions_min': [],
                'actions_mean': [],
                'actions_median': [],
                'actions_std': [],
                'actions_max':[],
            },
        }
        super().__init__(
            input_dir, output_dir,
            filename='actions.json',
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
                actions = []
                for episode in complete['hist_stats']['rewards_by_agent']:
                    for agent_rewards in episode.values():
                        actions.append(len(agent_rewards))
                self._aggregated_dataset['learning']['actions_min'].append(
                    np.nanmin(actions))
                self._aggregated_dataset['learning']['actions_median'].append(
                    np.nanmedian(actions))
                self._aggregated_dataset['learning']['actions_mean'].append(
                    np.nanmean(actions))
                self._aggregated_dataset['learning']['actions_std'].append(
                    np.nanstd(actions))
                self._aggregated_dataset['learning']['actions_max'].append(
                    np.nanmax(actions))

                # EVALUATION
                if 'evaluation' in complete:
                    complete['evaluation']['timesteps_total'] = complete['timesteps_total']
                    complete = complete['evaluation']

                    actions = []
                    for episode in complete['hist_stats']['rewards_by_agent']:
                        for agent_rewards in episode.values():
                            actions.append(len(agent_rewards))

                    self._aggregated_dataset['evaluation']['timesteps_total'].append(
                        complete['timesteps_total'])
                    self._aggregated_dataset['evaluation']['actions_min'].append(
                        np.nanmin(actions))
                    self._aggregated_dataset['evaluation']['actions_median'].append(
                        np.nanmedian(actions))
                    self._aggregated_dataset['evaluation']['actions_mean'].append(
                        np.nanmean(actions))
                    self._aggregated_dataset['evaluation']['actions_std'].append(
                        np.nanstd(actions))
                    self._aggregated_dataset['evaluation']['actions_max'].append(
                        np.nanmax(actions))

    def _generate_graphs(self):
        # LEARNING
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.errorbar(
            self._aggregated_dataset['learning']['timesteps_total'],
            self._aggregated_dataset['learning']['actions_mean'],
            yerr=self._aggregated_dataset['learning']['actions_std'],
            capsize=5, label='Mean [std]', fmt='-o')
        ax.plot(
            self._aggregated_dataset['learning']['timesteps_total'],
            self._aggregated_dataset['learning']['actions_min'], label='Min')
        ax.plot(
            self._aggregated_dataset['learning']['timesteps_total'],
            self._aggregated_dataset['learning']['actions_max'], label='Max')
        ax.plot(
            self._aggregated_dataset['learning']['timesteps_total'],
            self._aggregated_dataset['learning']['actions_median'], label='Median')
        ax.set(
            xlabel='Learning step', ylabel='Actions [#]', title='[L] Actions over time.')
        ax.legend(loc='best', ncol=4, shadow=True)
        ax.grid()
        fig.savefig('{}/learning.average_actions_over_learning.svg'.format(self._output_dir),
                    dpi=300, transparent=False, bbox_inches='tight')
        #plt.show()
        matplotlib.pyplot.close('all')

        # EVALUATION
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.errorbar(
            self._aggregated_dataset['evaluation']['timesteps_total'],
            self._aggregated_dataset['evaluation']['actions_mean'],
            yerr=self._aggregated_dataset['evaluation']['actions_std'],
            capsize=5, label='Mean [std]', fmt='-o')
        ax.plot(
            self._aggregated_dataset['evaluation']['timesteps_total'],
            self._aggregated_dataset['evaluation']['actions_min'], label='Min')
        ax.plot(
            self._aggregated_dataset['evaluation']['timesteps_total'],
            self._aggregated_dataset['evaluation']['actions_max'], label='Max')
        ax.plot(
            self._aggregated_dataset['evaluation']['timesteps_total'],
            self._aggregated_dataset['evaluation']['actions_median'], label='Median')
        ax.set(
            xlabel='Learning step', ylabel='Actions [#]', title='[E] Actions over time.')
        ax.legend(loc='best', ncol=4, shadow=True)
        ax.grid()
        fig.savefig('{}/evaluation.average_actions_over_learning.svg'.format(self._output_dir),
                    dpi=300, transparent=False, bbox_inches='tight')
        #plt.show()
        matplotlib.pyplot.close('all')

####################################################################################################

    # def average_actions_over_episodes_total(self):
    #     logger.info('Computing the average number of actions over the episodes.')
    #     x_coords = []
    #     mean_y = []
    #     median_y = []
    #     min_y = []
    #     max_y = []
    #     std_y = []
    #     episodes = 0
    #     with open(self.input, 'r') as jsonfile:
    #         counter = 0
    #         for row in tqdm(jsonfile): # enumerate cannot be used due to the size of the file
    #             complete = json.loads(row)
    #             if self.evaluation:
    #                 if 'evaluation' in complete:
    #                     complete = complete['evaluation']
    #                 else:
    #                     # evaluation stats requested but not present in the results
    #                     continue
    #             if 'rewards_by_agent' in complete['hist_stats']:
    #                 episodes += complete['episodes_this_iter']
    #                 x_coords.append(episodes)
    #                 _actions = []
    #                 for episode in complete['hist_stats']['rewards_by_agent']:
    #                     for agent_rewards in episode.values():
    #                         _actions.append(len(agent_rewards))
    #                 min_y.append(np.nanmin(_actions))
    #                 max_y.append(np.nanmax(_actions))
    #                 median_y.append(np.nanmedian(_actions))
    #                 mean_y.append(np.nanmean(_actions))
    #                 std_y.append(np.nanstd(_actions))
    #             else:
    #                 logger.critical('Missing stats in row %d', counter)
    #             counter += 1

    #     fig, ax = plt.subplots(figsize=(15, 10))
    #     ax.errorbar(x_coords, mean_y, yerr=std_y, capsize=5, label='Mean [std]', fmt='-o')
    #     ax.plot(x_coords, min_y, label='Min')
    #     ax.plot(x_coords, max_y, label='Max')
    #     ax.plot(x_coords, median_y, label='Median')
    #     ax.set(xlabel='Episodes', ylabel='Actions', title='Actions per episode')
    #     ax.legend(loc='best', ncol=4, shadow=True)
    #     ax.grid()
    #     fig.savefig('{}.actions_over_episodes.svg'.format(self.prefix),
    #                 dpi=300, transparent=False, bbox_inches='tight')
    #     # plt.show()
    #     matplotlib.pyplot.close('all')

####################################################################################################

if __name__ == '__main__':
    _main()

####################################################################################################
