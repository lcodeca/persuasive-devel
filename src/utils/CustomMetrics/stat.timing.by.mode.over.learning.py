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

from copy import deepcopy
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

WAITING_M = 402.17 / 60.0
LATENESS_M = 761.81 / 60.0

NUM_LATE = 595.43

ARRIVAL_H = 32715.66 / 3600.0
DEPARTURE_H = 31613.13 / 3600.0
TRAVEL_TIME_M = 1102.53 / 60.0

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

    TimingByMode(config.input_dir, config.output_dir).generate()
    logging.info('Done')

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler.disable()
        results = io.StringIO()
        pstats.Stats(profiler, stream=results).sort_stats('cumulative').print_stats(50)
        logging.info('Profiler: \n%s', pformat(results.getvalue()))
    ## ========================              PROFILER              ======================== ##

####################################################################################################

class TimingByMode(GenericGraphMaker):

    def __init__(self, input_dir, output_dir):
        self._default_modes = ['passenger', 'public', 'walk', 'bicycle', 'ptw']
        metrics_by_mode = {
            'agents_mean': [],
            'agents_median': [],
            'agents_std': [],
            #######################
            'ownership_mean': [],
            #######################
            'reward_mean': [],
            'reward_median': [],
            'reward_std': [],
            #######################
            'too_late_mean': [],
            'too_late_median': [],
            'too_late_std': [],
            #######################
            'waiting_mean': [],
            'waiting_median': [],
            'waiting_std': [],
            #######################
            'lateness_mean': [],
            'lateness_median': [],
            'lateness_std': [],
            #######################
            'departure_mean': [],
            'departure_median': [],
            'departure_std': [],
            #######################
            'ttime_mean': [],
            'ttime_median': [],
            'ttime_std': [],
            #######################
            'arrival_mean': [],
            'arrival_median': [],
            'arrival_std': [],
            #######################
        }
        _default = {
            'action-to-mode': {
                0: 'wait',
                1: 'passenger',
                2: 'public',
                3: 'walk',
                4: 'bicycle',
                5: 'ptw'
            },
            'learning': dict(),
            'evaluation': dict(),
        }
        for mode in self._default_modes:
            _default['learning'][mode] = deepcopy(metrics_by_mode)
            _default['evaluation'][mode] = deepcopy(metrics_by_mode)
        _default['learning']['timesteps_total'] = []
        _default['evaluation']['timesteps_total'] = []

        super().__init__(
            input_dir, output_dir,
            filename='timing-by-mode.json',
            default=_default)

    def _find_last_metric(self):
        return len(self._aggregated_dataset['learning']['timesteps_total'])

    def _aggregate_metrics_helper(self, metrics, tag):
        self._aggregated_dataset[tag]['timesteps_total'].append(
            metrics['timesteps_total'])

        info_by_episode = metrics['hist_stats']['info_by_agent']
        last_action_by_agent = metrics['hist_stats']['last_action_by_agent']
        rewards_by_agent = metrics['hist_stats']['rewards_by_agent']

        avg_agents = collections.defaultdict(list)
        avg_arrival_s = collections.defaultdict(list)
        avg_departure_s = collections.defaultdict(list)
        avg_lateness_s = collections.defaultdict(list)
        avg_ownership = collections.defaultdict(list)
        avg_reward = collections.defaultdict(list)
        avg_too_late = collections.defaultdict(list)
        avg_travel_time_s = collections.defaultdict(list)
        avg_waiting_s = collections.defaultdict(list)

        for pos, episode in enumerate(info_by_episode):
            agents = collections.defaultdict(int)
            arrival_s = collections.defaultdict(list)
            departure_s = collections.defaultdict(list)
            lateness_s = collections.defaultdict(list)
            ownership = collections.defaultdict(int)
            reward = collections.defaultdict(list)
            too_late = collections.defaultdict(int)
            travel_time_s = collections.defaultdict(list)
            waiting_s = collections.defaultdict(list)

            for agent, info in episode.items():
                if np.isnan(info['arrival']):
                    # Nothing to do with agents that waited for too long.
                    continue

                reward[info['mode']].append(np.sum(rewards_by_agent[pos][agent]))

                departure_s[info['mode']].append(info['departure'])
                arrival_s[info['mode']].append(info['arrival'])
                travel_time_s[info['mode']].append(info['arrival']-info['departure'])
                agents[info['mode']] += 1

                if info['arrival'] > info['init']['exp-arrival']:
                    ## LATE
                    too_late[info['mode']] += 1
                    lateness_s[info['mode']].append(info['arrival']-info['init']['exp-arrival'])
                else:
                    ## WAITING
                    waiting_s[info['mode']].append(info['init']['exp-arrival']-info['arrival'])

                if 'ownership' in info['init']:
                    for mode, val in info['init']['ownership'].items():
                        if val:
                            ownership[mode] += 1

            for mode, values in agents.items():
                avg_agents[mode].append(values)
            for mode, values in arrival_s.items():
                avg_arrival_s[mode].append(np.nanmean(values))
            for mode, values in departure_s.items():
                avg_departure_s[mode].append(np.nanmean(values))
            for mode, values in lateness_s.items():
                avg_lateness_s[mode].append(np.nanmean(values))
            for mode, values in ownership.items():
                avg_ownership[mode].append(values)
            for mode, values in reward.items():
                avg_reward[mode].append(np.nanmean(values))
            for mode, values in too_late.items():
                avg_too_late[mode].append(values)
            for mode, values in travel_time_s.items():
                avg_travel_time_s[mode].append(np.nanmean(values))
            for mode, values in waiting_s.items():
                avg_waiting_s[mode].append(np.nanmean(values))

        for mode in self._default_modes:
            # REWARD
            self._aggregated_dataset[tag][mode]['agents_mean'].append(np.nanmean(avg_agents[mode]))
            self._aggregated_dataset[tag][mode]['agents_median'].append(np.nanmedian(avg_agents[mode]))
            self._aggregated_dataset[tag][mode]['agents_std'].append(np.nanstd(avg_agents[mode]))
            # REWARD
            self._aggregated_dataset[tag][mode]['reward_mean'].append(np.nanmean(avg_reward[mode]))
            self._aggregated_dataset[tag][mode]['reward_median'].append(np.nanmedian(avg_reward[mode]))
            self._aggregated_dataset[tag][mode]['reward_std'].append(np.nanstd(avg_reward[mode]))
            ## TOO LATE
            self._aggregated_dataset[tag][mode]['too_late_mean'].append(np.nanmean(avg_too_late[mode]))
            self._aggregated_dataset[tag][mode]['too_late_median'].append(np.nanmedian(avg_too_late[mode]))
            self._aggregated_dataset[tag][mode]['too_late_std'].append(np.nanstd(avg_too_late[mode]))
            ## WAITING
            self._aggregated_dataset[tag][mode]['waiting_mean'].append(np.nanmean(avg_waiting_s[mode])/60)
            self._aggregated_dataset[tag][mode]['waiting_median'].append(np.nanmedian(avg_waiting_s[mode])/60)
            self._aggregated_dataset[tag][mode]['waiting_std'].append(np.nanstd(avg_waiting_s[mode])/60)
            ## TOO LATE
            self._aggregated_dataset[tag][mode]['lateness_mean'].append(np.nanmean(avg_lateness_s[mode])/60)
            self._aggregated_dataset[tag][mode]['lateness_median'].append(np.nanmedian(avg_lateness_s[mode])/60)
            self._aggregated_dataset[tag][mode]['lateness_std'].append(np.nanstd(avg_lateness_s[mode])/60)
            ## DEARTURE
            self._aggregated_dataset[tag][mode]['departure_mean'].append(np.nanmean(avg_departure_s[mode])/3600)
            self._aggregated_dataset[tag][mode]['departure_median'].append(np.nanmedian(avg_departure_s[mode])/3600)
            self._aggregated_dataset[tag][mode]['departure_std'].append(np.nanstd(avg_departure_s[mode])/3600)
            ## TRAVEL TIME
            self._aggregated_dataset[tag][mode]['ttime_mean'].append(np.nanmean(avg_travel_time_s[mode])/60)
            self._aggregated_dataset[tag][mode]['ttime_median'].append(np.nanmedian(avg_travel_time_s[mode])/60)
            self._aggregated_dataset[tag][mode]['ttime_std'].append(np.nanstd(avg_travel_time_s[mode])/60)
            ## ARRIVAL
            self._aggregated_dataset[tag][mode]['arrival_mean'].append(np.nanmean(avg_arrival_s[mode])/3600)
            self._aggregated_dataset[tag][mode]['arrival_median'].append(np.nanmedian(avg_arrival_s[mode])/3600)
            self._aggregated_dataset[tag][mode]['arrival_std'].append(np.nanstd(avg_arrival_s[mode])/3600)
            ## OWNERSHIP
            self._aggregated_dataset[tag][mode]['ownership_mean'].append(np.nanmean(avg_ownership[mode]))

    def _aggregate_metrics(self, files):
        for filename in tqdm(files):
            # print(filename)
            with open(os.path.join(self._input_dir, filename), 'r') as jsonfile:
                complete = json.load(jsonfile)

                if 'action-to-mode' in complete['config']['env_config']['agent_init']:
                    self._aggregated_dataset['action-to-mode'] = \
                        complete['config']['env_config']['agent_init']['action-to-mode']
                    self._aggregated_dataset['action-to-mode'][0] = 'wait'

                # LEARNING
                self._aggregate_metrics_helper(complete, 'learning')

                # EVALUATION
                if 'evaluation' in complete:
                    complete['evaluation']['timesteps_total'] = complete['timesteps_total']
                    complete = complete['evaluation']
                    self._aggregate_metrics_helper(complete, 'evaluation')

    def _generate_graphs_helper(self, tag, lbl):
        fig, axs = plt.subplots(
            8, len(self._default_modes),
            figsize=(55, 35),
            constrained_layout=True) # , sharex=True
        fig.suptitle('{} Aggregated Timing by Mode over Learning'.format(lbl))

        for pos, mode in enumerate(self._default_modes):

            ## AGENTS
            axs[0][pos].axhline(y=0, linestyle=':')
            axs[0][pos].errorbar(
                self._aggregated_dataset[tag]['timesteps_total'],
                self._aggregated_dataset[tag][mode]['agents_mean'],
                yerr=self._aggregated_dataset[tag][mode]['agents_std'],
                capsize=5, label='Mean [std]', fmt='-o')
            axs[0][pos].plot(
                self._aggregated_dataset[tag]['timesteps_total'],
                self._aggregated_dataset[tag][mode]['agents_median'], label='Median')
            axs[0][pos].plot(
                self._aggregated_dataset[tag]['timesteps_total'],
                self._aggregated_dataset[tag][mode]['ownership_mean'], label='Ownership')
            axs[0][pos].set(ylabel='Agents [#]', title=mode.upper())
            axs[0][pos].legend(ncol=4, loc='best', shadow=True)
            axs[0][pos].grid()

            ## REWARDS
            axs[1][pos].axhline(y=0, linestyle=':')
            axs[1][pos].errorbar(
                self._aggregated_dataset[tag]['timesteps_total'],
                self._aggregated_dataset[tag][mode]['reward_mean'],
                yerr=self._aggregated_dataset[tag][mode]['reward_std'],
                capsize=5, label='Mean [std]', fmt='-o')
            axs[1][pos].plot(
                self._aggregated_dataset[tag]['timesteps_total'],
                self._aggregated_dataset[tag][mode]['reward_median'], label='Median')
            axs[1][pos].set(ylabel='Reward [#]', title='Reward over time.')
            axs[1][pos].legend(ncol=3, loc='best', shadow=True)
            axs[1][pos].grid()

            ## TOO LATE
            axs[2][pos].axhline(y=0, linestyle=':')
            # axs[2][pos].axhline(y=NUM_LATE, color='r', label='Baseline')
            axs[2][pos].errorbar(
                self._aggregated_dataset[tag]['timesteps_total'],
                self._aggregated_dataset[tag][mode]['too_late_mean'],
                yerr=self._aggregated_dataset[tag][mode]['too_late_std'],
                capsize=5, label='Mean [std]', fmt='-o')
            axs[2][pos].plot(
                self._aggregated_dataset[tag]['timesteps_total'],
                self._aggregated_dataset[tag][mode]['too_late_median'], label='Median')
            axs[2][pos].set(ylabel='Late [#]', title='Arrived too late over time.')
            axs[2][pos].legend(ncol=3, loc='best', shadow=True)
            axs[2][pos].grid()

            ## LATENESS
            axs[3][pos].axhline(y=0, linestyle=':')
            axs[3][pos].axhline(y=LATENESS_M, color='r', label='Baseline')
            axs[3][pos].errorbar(
                self._aggregated_dataset[tag]['timesteps_total'],
                self._aggregated_dataset[tag][mode]['lateness_mean'],
                yerr=self._aggregated_dataset[tag][mode]['lateness_std'],
                capsize=5, label='Mean [std]', fmt='-o')
            axs[3][pos].plot(
                self._aggregated_dataset[tag]['timesteps_total'],
                self._aggregated_dataset[tag][mode]['lateness_median'], label='Median')
            axs[3][pos].set(ylabel='Time [m]', title='Lateness over time.')
            axs[3][pos].legend(ncol=3, loc='best', shadow=True)
            axs[3][pos].grid()

            ## WAITING
            axs[4][pos].axhline(y=0, linestyle=':')
            axs[4][pos].axhline(y=WAITING_M, color='r', label='Baseline')
            axs[4][pos].errorbar(
                self._aggregated_dataset[tag]['timesteps_total'],
                self._aggregated_dataset[tag][mode]['waiting_mean'],
                yerr=self._aggregated_dataset[tag][mode]['waiting_std'],
                capsize=5, label='Mean [std]', fmt='-o')
            axs[4][pos].plot(
                self._aggregated_dataset[tag]['timesteps_total'],
                self._aggregated_dataset[tag][mode]['waiting_median'], label='Median')
            axs[4][pos].set(ylabel='Time [m]', title='Waiting over time.')
            axs[4][pos].legend(ncol=3, loc='best', shadow=True)
            axs[4][pos].grid()

            ## DEPARTURE
            axs[5][pos].axhline(y=9.0, linestyle=':')
            axs[5][pos].axhline(y=DEPARTURE_H, color='r', label='Baseline')
            axs[5][pos].errorbar(
                self._aggregated_dataset[tag]['timesteps_total'],
                self._aggregated_dataset[tag][mode]['departure_mean'],
                yerr=self._aggregated_dataset[tag][mode]['departure_std'],
                capsize=5, label='Mean [std]', fmt='-o')
            axs[5][pos].plot(
                self._aggregated_dataset[tag]['timesteps_total'],
                self._aggregated_dataset[tag][mode]['departure_median'], label='Median')
            axs[5][pos].set(ylabel='Time [h]', title='Departure over time.')
            axs[5][pos].legend(ncol=3, loc='best', shadow=True)
            axs[5][pos].grid()

            ## ARRIVAL
            axs[6][pos].axhline(9.0, linestyle=':')
            axs[6][pos].axhline(y=ARRIVAL_H, color='r', label='Baseline')
            axs[6][pos].errorbar(
                self._aggregated_dataset[tag]['timesteps_total'],
                self._aggregated_dataset[tag][mode]['arrival_mean'],
                yerr=self._aggregated_dataset[tag][mode]['arrival_std'],
                capsize=5, label='Mean [std]', fmt='-o')
            axs[6][pos].plot(
                self._aggregated_dataset[tag]['timesteps_total'],
                self._aggregated_dataset[tag][mode]['arrival_median'], label='Median')
            axs[6][pos].set(ylabel='Time [h]', title='Arrival over time.')
            axs[6][pos].legend(ncol=3, loc='best', shadow=True)
            axs[6][pos].grid()

            ## TRAVEL TIME
            axs[7][pos].axhline(y=TRAVEL_TIME_M, color='r', label='Baseline')
            axs[7][pos].errorbar(
                self._aggregated_dataset[tag]['timesteps_total'],
                self._aggregated_dataset[tag][mode]['ttime_mean'],
                yerr=self._aggregated_dataset[tag][mode]['ttime_std'],
                capsize=5, label='Mean [std]', fmt='-o')
            axs[7][pos].plot(
                self._aggregated_dataset[tag]['timesteps_total'],
                self._aggregated_dataset[tag][mode]['ttime_median'], label='Median')
            axs[7][pos].set(ylabel='Time [m]', title='Travel time over time.')
            axs[7][pos].legend(ncol=3, loc='best', shadow=True)
            axs[7][pos].grid()

        fig.savefig('{}/{}.timing_by_mode_over_learning.svg'.format(self._output_dir, tag),
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
