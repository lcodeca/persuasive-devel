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

    Achievements(config.input_dir, config.output_dir).generate()
    logging.info('Done')

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler.disable()
        results = io.StringIO()
        pstats.Stats(profiler, stream=results).sort_stats('cumulative').print_stats(50)
        logging.info('Profiler: \n%s', pformat(results.getvalue()))
    ## ========================              PROFILER              ======================== ##

####################################################################################################

class Achievements(GenericGraphMaker):

    def __init__(self, input_dir, output_dir):
        self._default_by_choice = [
            '_choice', '_choice_reward', '_choice_num_late', '_choice_lateness', '_choice_waiting',
            '_choice_walk', '_choice_bicycle', '_choice_public', '_choice_passenger', '_choice_ptw']
        self._choices_tags = ['1st', '2nd', '3rd', 'rest']

        _default = {
            'learning': {
                'timesteps_total': [],
            },
            'evaluation': {
                'timesteps_total': [],
            },
        }
        for _choices in self._choices_tags:
            for _metric in self._default_by_choice:
                for _tag in ['learning', 'evaluation']:
                    _default[_tag]['{}{}'.format(_choices, _metric)] = []

        super().__init__(
            input_dir, output_dir,
            filename='achievements.json',
            default=_default)

    def _find_last_metric(self):
        return len(self._aggregated_dataset['learning']['timesteps_total'])

    def _learning_step_res_helper(self):
        _ret = {}
        for _choices in self._choices_tags:
            for _metric in self._default_by_choice:
                _ret['{}{}'.format(_choices, _metric)] = []
        return _ret

    def _episode_res_helper(self):
        _ret = {}
        for _choices in self._choices_tags:
            for _metric in self._default_by_choice:
                _tag = _metric.split('_')[-1]
                if _tag in ['choice', 'late', 'walk', 'bicycle', 'public', 'passenger', 'ptw']:
                    _ret['{}{}'.format(_choices, _metric)] = 0
                elif _tag in ['reward', 'lateness', 'waiting']:
                    _ret['{}{}'.format(_choices, _metric)] = []
                else:
                    raise Exception('_episode_res_helper: {} - {}'.format(_choices, _metric))
        return _ret

    def _aggregate_episodes_helper(self):
        """ Based on
        - self._learning_step_metrics
        - self._curr_metrics
        """
        for _metric, _values in self._curr_metrics.items():
            _tag = _metric.split('_')[-1]
            if _tag in ['choice', 'late', 'walk', 'bicycle', 'public', 'passenger', 'ptw']:
                self._learning_step_metrics[_metric].append(_values)
            elif _tag in ['reward', 'lateness', 'waiting']:
                self._learning_step_metrics[_metric].append(np.mean(_values))
            else:
                raise Exception('_aggregate_episodes_helper: {} - {}'.format(_metric, _values))

    def _choice_res_helper(self, choice, info, reward):
        self._curr_metrics['{}_choice'.format(choice)] += 1
        self._curr_metrics['{}_choice_{}'.format(choice, info['mode'])] += 1
        self._curr_metrics['{}_choice_reward'.format(choice)].append(reward)
        if np.isnan(info['wait']):
            # late
            print('LATE:', choice, reward, info['mode'])
            self._curr_metrics['{}_choice_num_late'.format(choice)] += 1
            self._curr_metrics['{}_choice_lateness'.format(choice)].append(
                (info['arrival']-info['init']['exp-arrival'])/60.0)
        else:
            # print('wait', choice, reward, info['mode'], info['wait'])
            self._curr_metrics['{}_choice_waiting'.format(choice)].append(
                info['wait']/60.0)

    def _aggregate_learning_steps_helper(self, tag):
        """ Based on
        - self._learning_step_metrics
        - self._aggregated_dataset
        """
        for _metric, _values in self._learning_step_metrics.items():
            self._aggregated_dataset[tag][_metric].append(np.nanmean(_values))

    def _aggregate_metrics_helper(self, tag, metrics):

        rewards_by_agent = metrics['hist_stats']['rewards_by_agent']

        self._aggregated_dataset[tag]['timesteps_total'].append(
            metrics['timesteps_total'])

        self._learning_step_metrics = self._learning_step_res_helper()

        for pos, episode in enumerate(metrics['hist_stats']['info_by_agent']):
            self._curr_metrics = self._episode_res_helper()

            for agent, info in episode.items():
                if 'preferences' not in info['init']:
                    continue
                pref = sorted([(p, m) for m, p in info['init']['preferences'].items()])
                if not np.isnan(info['arrival']):
                    # A choice was made.
                    if info['mode'] == pref[0][1]:
                        self._choice_res_helper('1st', info, np.sum(rewards_by_agent[pos][agent]))
                    elif info['mode'] == pref[1][1]:
                        self._choice_res_helper('2nd', info, np.sum(rewards_by_agent[pos][agent]))
                    elif info['mode'] == pref[2][1]:
                        self._choice_res_helper('3rd', info, np.sum(rewards_by_agent[pos][agent]))
                    else:
                        self._choice_res_helper('rest', info, np.sum(rewards_by_agent[pos][agent]))

            self._aggregate_episodes_helper()

        self._aggregate_learning_steps_helper(tag)

    def _aggregate_metrics(self, files):
        for filename in tqdm(files):
            # print(filename)
            with open(os.path.join(self._input_dir, filename), 'r') as jsonfile:
                complete = json.load(jsonfile)

                # LEARNING
                self._aggregate_metrics_helper('learning', complete)

                # EVALUATION
                if 'evaluation' in complete:
                    complete['evaluation']['timesteps_total'] = complete['timesteps_total']
                    complete = complete['evaluation']
                    self._aggregate_metrics_helper('evaluation', complete)

    def _generate_subplot_helper(self, axs, tag,  col, name, string):
        ## ## ##
        axs[0][col].axhline(y=0, linestyle=':')
        axs[0][col].plot(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['{}_choice'.format(name)], label='Mean')
        for mode in ['walk', 'bicycle', 'public', 'passenger', 'ptw']:
            axs[0][col].plot(
                self._aggregated_dataset[tag]['timesteps_total'],
                self._aggregated_dataset[tag]['{}_choice_{}'.format(name, mode)], label=mode)
        axs[0][col].set(ylabel='Mean Agents [#]', title='{} Choice'.format(string))
        axs[0][col].legend(ncol=3, loc='best', shadow=True)
        axs[0][col].grid()
        ## ## ##
        axs[1][col].axhline(y=0, linestyle=':')
        axs[1][col].plot(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['{}_choice_num_late'.format(name)], label='Mean')
        axs[1][col].set(ylabel='Mean Late Agents [#]')
        # axs[1][0].legend(ncol=3, loc='best', shadow=True)
        axs[1][col].grid()
        ## ## ##
        axs[2][col].axhline(y=0, linestyle=':')
        axs[2][col].plot(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['{}_choice_lateness'.format(name)], label='Mean')
        axs[2][col].set(ylabel='Mean Lateness [min]')
        # axs[2][col].legend(ncol=3, loc='best', shadow=True)
        axs[2][col].grid()
        ## ## ##
        axs[3][col].axhline(y=0, linestyle=':')
        axs[3][col].plot(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['{}_choice_waiting'.format(name)], label='Mean')
        axs[3][col].set(ylabel='Mean Waiting [min]')
        # axs[3][col].legend(ncol=3, loc='best', shadow=True)
        axs[3][col].grid()
        ## ## ##
        axs[4][col].axhline(y=0, linestyle=':')
        axs[4][col].plot(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['{}_choice_reward'.format(name)], label='Mean')
        axs[4][col].set(xlabel='Learning step', ylabel='Mean Reward [#]')
        # axs[4][col].legend(ncol=3, loc='best', shadow=True)
        axs[4][col].grid()
        ########################################################################

    def _generate_graphs_helper(self, tag, lbl):
        fig, axs = plt.subplots(5, 4, figsize=(45, 25),
                                constrained_layout=True, sharey='row') # sharex='col',
        fig.suptitle('{} Aggregated Achievements over Learning'.format(lbl))

        self._generate_subplot_helper(axs, tag, col=0, name='1st', string='First')
        self._generate_subplot_helper(axs, tag, col=1, name='2nd', string='Second')
        self._generate_subplot_helper(axs, tag, col=2, name='3rd', string='Third')
        self._generate_subplot_helper(axs, tag, col=3, name='rest', string='Other')

        fig.savefig('{}/{}.achievements_over_learning.svg'.format(self._output_dir, tag),
                    dpi=300, transparent=False, bbox_inches='tight')
        # plt.show()
        matplotlib.pyplot.close('all')

    def _generate_graphs(self):
        self._generate_graphs_helper('learning', '[L]')
        self._generate_graphs_helper('evaluation', '[E]')

####################################################################################################

if __name__ == '__main__':
    _main()

####################################################################################################
