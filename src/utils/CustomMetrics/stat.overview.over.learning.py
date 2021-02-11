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

    Overview(config.input_dir, config.output_dir).generate()
    logging.info('Done')

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler.disable()
        results = io.StringIO()
        pstats.Stats(profiler, stream=results).sort_stats('cumulative').print_stats(50)
        logging.info('Profiler: \n%s', pformat(results.getvalue()))
    ## ========================              PROFILER              ======================== ##

####################################################################################################

class Overview(GenericGraphMaker):

    def __init__(self, input_dir, output_dir):
        metrics = {
            'timesteps_total': [],
            #######################
            'reward_mean': [],
            'reward_median': [],
            'reward_std': [],
            #######################
            'missing_mean': [],
            'missing_median': [],
            'missing_std': [],
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
            'm_wait_mean': [],
            'm_wait_median': [],
            'm_wait_std': [],
            #######################
            'm_walk_mean': [],
            'm_walk_median': [],
            'm_walk_std': [],
            #######################
            'm_bicycle_mean': [],
            'm_bicycle_median': [],
            'm_bicycle_std': [],
            #######################
            'm_pt_mean': [],
            'm_pt_median': [],
            'm_pt_std': [],
            #######################
            'm_car_mean': [],
            'm_car_median': [],
            'm_car_std': [],
            #######################
            'm_ptw_mean': [],
            'm_ptw_median': [],
            'm_ptw_std': [],
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
            'learning': deepcopy(metrics),
            'evaluation': deepcopy(metrics),
        }
        super().__init__(
            input_dir, output_dir,
            filename='overview.json',
            default=_default)
        self._default_mode_to_metric = {
            'wait': 'm_wait_',
            'passenger': 'm_car_',
            'public': 'm_pt_',
            'walk': 'm_walk_',
            'bicycle': 'm_bicycle_',
            'ptw': 'm_ptw_'
        }

    def _find_last_metric(self):
        return len(self._aggregated_dataset['learning']['timesteps_total'])

    def _aggregate_metrics_helper(self, metrics, tag):
        self._aggregated_dataset[tag]['timesteps_total'].append(
            metrics['timesteps_total'])

        # REWARD
        self._aggregated_dataset[tag]['reward_median'].append(
            np.nanmedian(metrics['hist_stats']['policy_unique_reward']))
        self._aggregated_dataset[tag]['reward_mean'].append(
            np.nanmean(metrics['hist_stats']['policy_unique_reward']))
        self._aggregated_dataset[tag]['reward_std'].append(
            np.nanstd(metrics['hist_stats']['policy_unique_reward']))

        info_by_episode = metrics['hist_stats']['info_by_agent']
        last_action_by_agent = metrics['hist_stats']['last_action_by_agent']

        avg_arrival_s = []
        avg_departure_s = []
        avg_lateness_s = []
        avg_waiting_s = []
        avg_travel_time_s = []
        avg_missing = []
        avg_too_late = []
        avg_modes = collections.defaultdict(list)

        for pos, episode in enumerate(info_by_episode):
            arrival_s = []
            departure_s = []
            lateness_s = []
            waiting_s = []
            travel_time_s = []
            missing = 0
            too_late = 0
            modes = collections.defaultdict(int)
            for agent, info in episode.items():
                if np.isnan(info['arrival']):
                    # MISSING
                    missing += 1
                    if last_action_by_agent[pos][agent] == 0:
                        modes[last_action_by_agent[pos][agent]] += 1
                else:
                    departure_s.append(info['departure'])
                    arrival_s.append(info['arrival'])
                    travel_time_s.append(info['arrival']-info['departure'])
                    modes[last_action_by_agent[pos][agent]] += 1
                    if info['arrival'] > info['init']['exp-arrival']:
                        ## LATE
                        too_late += 1
                        lateness_s.append(info['arrival']-info['init']['exp-arrival'])
                    else:
                        waiting_s.append(info['init']['exp-arrival']-info['arrival'])
            avg_arrival_s.append(np.nanmean(arrival_s))
            avg_departure_s.append(np.nanmean(departure_s))
            avg_lateness_s.append(np.nanmean(lateness_s))
            avg_waiting_s.append(np.nanmean(waiting_s))
            avg_travel_time_s.append(np.nanmean(travel_time_s))
            avg_missing.append(missing)
            avg_too_late.append(too_late)
            for mode, val in modes.items():
                avg_modes[mode].append(val)

        ## MISSING
        self._aggregated_dataset[tag]['missing_mean'].append(np.nanmean(avg_missing))
        self._aggregated_dataset[tag]['missing_median'].append(np.nanmedian(avg_missing))
        self._aggregated_dataset[tag]['missing_std'].append(np.nanstd(avg_missing))
        ## TOO LATE
        self._aggregated_dataset[tag]['too_late_mean'].append(np.nanmean(avg_too_late))
        self._aggregated_dataset[tag]['too_late_median'].append(np.nanmedian(avg_too_late))
        self._aggregated_dataset[tag]['too_late_std'].append(np.nanstd(avg_too_late))
        ## WAITING
        self._aggregated_dataset[tag]['waiting_mean'].append(np.nanmean(avg_waiting_s)/60)
        self._aggregated_dataset[tag]['waiting_median'].append(np.nanmedian(avg_waiting_s)/60)
        self._aggregated_dataset[tag]['waiting_std'].append(np.nanstd(avg_waiting_s)/60)
        ## TOO LATE
        self._aggregated_dataset[tag]['lateness_mean'].append(np.nanmean(avg_lateness_s)/60)
        self._aggregated_dataset[tag]['lateness_median'].append(np.nanmedian(avg_lateness_s)/60)
        self._aggregated_dataset[tag]['lateness_std'].append(np.nanstd(avg_lateness_s)/60)
        ## DEARTURE
        self._aggregated_dataset[tag]['departure_mean'].append(np.nanmean(avg_departure_s)/3600)
        self._aggregated_dataset[tag]['departure_median'].append(np.nanmedian(avg_departure_s)/3600)
        self._aggregated_dataset[tag]['departure_std'].append(np.nanstd(avg_departure_s)/3600)
        ## TRAVEL TIME
        self._aggregated_dataset[tag]['ttime_mean'].append(np.nanmean(avg_travel_time_s)/60)
        self._aggregated_dataset[tag]['ttime_median'].append(np.nanmedian(avg_travel_time_s)/60)
        self._aggregated_dataset[tag]['ttime_std'].append(np.nanstd(avg_travel_time_s)/60)
        ## ARRIVAL
        self._aggregated_dataset[tag]['arrival_mean'].append(np.nanmean(avg_arrival_s)/3600)
        self._aggregated_dataset[tag]['arrival_median'].append(np.nanmedian(avg_arrival_s)/3600)
        self._aggregated_dataset[tag]['arrival_std'].append(np.nanstd(avg_arrival_s)/3600)

        ## All the MODES
        for _current_mode, _current_metric in self._default_mode_to_metric.items():
            # print(_current_mode, _current_metric, avg_modes[self._mode_to_action[_current_mode]])
            self._aggregated_dataset[tag]['{}mean'.format(_current_metric)].append(
                np.nanmean(avg_modes[self._mode_to_action[_current_mode]]))
            self._aggregated_dataset[tag]['{}median'.format(_current_metric)].append(
                np.nanmedian(avg_modes[self._mode_to_action[_current_mode]]))
            self._aggregated_dataset[tag]['{}std'.format(_current_metric)].append(
                np.nanstd(avg_modes[self._mode_to_action[_current_mode]]))

    def _aggregate_metrics(self, files):
        for filename in tqdm(files):
            # print(filename)
            with open(os.path.join(self._input_dir, filename), 'r') as jsonfile:
                complete = json.load(jsonfile)

                if 'action-to-mode' in complete['config']['env_config']['agent_init']:
                    self._aggregated_dataset['action-to-mode'] = \
                        complete['config']['env_config']['agent_init']['action-to-mode']
                    self._aggregated_dataset['action-to-mode'][0] = 'wait'

                self._mode_to_action = {}
                for action, mode in self._aggregated_dataset['action-to-mode'].items():
                    self._mode_to_action[mode] = int(action)
                for mode in self._default_mode_to_metric:
                    if mode not in self._mode_to_action:
                        self._mode_to_action[mode] = -666 # cause i'm faily sure it's an
                                                          # impossible action value in TF

                # LEARNING
                self._aggregate_metrics_helper(complete, 'learning')

                # EVALUATION
                if 'evaluation' in complete:
                    complete['evaluation']['timesteps_total'] = complete['timesteps_total']
                    complete = complete['evaluation']
                    self._aggregate_metrics_helper(complete, 'evaluation')

    def _generate_graphs_helper(self, tag, lbl):
        fig, axs = plt.subplots(5, 3, figsize=(45, 25), constrained_layout=True) # , sharex=True
        fig.suptitle('{} Aggregated Overview over Learning'.format(lbl))

        ## REWARDS
        axs[0][0].axhline(y=0, linestyle=':')
        axs[0][0].errorbar(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['reward_mean'],
            yerr=self._aggregated_dataset[tag]['reward_std'],
            capsize=5, label='Mean [std]', fmt='-o')
        axs[0][0].plot(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['reward_median'], label='Median')
        axs[0][0].set(ylabel='Reward [#]', title='Reward over time.')
        axs[0][0].legend(ncol=3, loc='best', shadow=True)
        axs[0][0].grid()
        ## ## ##
        axs[1][0].axhline(y=0, linestyle=':')
        axs[1][0].errorbar(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['reward_mean'],
            yerr=self._aggregated_dataset[tag]['reward_std'],
            capsize=5, label='Mean [std]', fmt='-o')
        axs[1][0].plot(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['reward_median'], label='Median')
        axs[1][0].set_ylim(min(self._aggregated_dataset[tag]['reward_mean']) - 10,
                           max(self._aggregated_dataset[tag]['reward_mean']) + 10)
        axs[1][0].set(ylabel='Reward [#]', title='Zoomed Reward over time.')
        axs[1][0].legend(ncol=3, loc='best', shadow=True)
        axs[1][0].grid()

        ## MISSING
        axs[0][1].axhline(y=0, linestyle=':')
        axs[0][1].errorbar(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['missing_mean'],
            yerr=self._aggregated_dataset[tag]['missing_std'],
            capsize=5, label='Mean [std]', fmt='-o')
        axs[0][1].plot(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['missing_median'], label='Median')
        axs[0][1].set(ylabel='Missing [#]', title='Missing over time.')
        axs[0][1].legend(ncol=3, loc='best', shadow=True)
        axs[0][1].grid()

        ## TOO LATE
        axs[0][2].axhline(y=0, linestyle=':')
        axs[0][2].axhline(y=NUM_LATE, color='r', label='Baseline')
        axs[0][2].errorbar(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['too_late_mean'],
            yerr=self._aggregated_dataset[tag]['too_late_std'],
            capsize=5, label='Mean [std]', fmt='-o')
        axs[0][2].plot(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['too_late_median'], label='Median')
        axs[0][2].set(ylabel='Late [#]', title='Arrived too late over time.')
        axs[0][2].legend(ncol=3, loc='best', shadow=True)
        axs[0][2].grid()

        ## WAITING
        axs[1][1].axhline(y=0, linestyle=':')
        axs[1][1].axhline(y=WAITING_M, color='r', label='Baseline')
        axs[1][1].errorbar(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['waiting_mean'],
            yerr=self._aggregated_dataset[tag]['waiting_std'],
            capsize=5, label='Mean [std]', fmt='-o')
        axs[1][1].plot(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['waiting_median'], label='Median')
        axs[1][1].set(ylabel='Time [m]', title='Waiting over time.')
        axs[1][1].legend(ncol=3, loc='best', shadow=True)
        axs[1][1].grid()

        ## LATENESS
        axs[1][2].axhline(y=0, linestyle=':')
        axs[1][2].axhline(y=LATENESS_M, color='r', label='Baseline')
        axs[1][2].errorbar(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['lateness_mean'],
            yerr=self._aggregated_dataset[tag]['lateness_std'],
            capsize=5, label='Mean [std]', fmt='-o')
        axs[1][2].plot(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['lateness_median'], label='Median')
        axs[1][2].set(ylabel='Time [m]', title='Lateness over time.')
        axs[1][2].legend(ncol=3, loc='best', shadow=True)
        axs[1][2].grid()

        ## DEPARTURE
        axs[2][0].axhline(y=9.0, linestyle=':')
        axs[2][0].axhline(y=DEPARTURE_H, color='r', label='Baseline')
        axs[2][0].errorbar(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['departure_mean'],
            yerr=self._aggregated_dataset[tag]['departure_std'],
            capsize=5, label='Mean [std]', fmt='-o')
        axs[2][0].plot(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['departure_median'], label='Median')
        axs[2][0].set(ylabel='Time [h]', title='Departure over time.')
        axs[2][0].legend(ncol=3, loc='best', shadow=True)
        axs[2][0].grid()

        ## ARRIVAL
        axs[2][1].axhline(9.0, linestyle=':')
        axs[2][1].axhline(y=ARRIVAL_H, color='r', label='Baseline')
        axs[2][1].errorbar(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['arrival_mean'],
            yerr=self._aggregated_dataset[tag]['arrival_std'],
            capsize=5, label='Mean [std]', fmt='-o')
        axs[2][1].plot(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['arrival_median'], label='Median')
        axs[2][1].set(ylabel='Time [h]', title='Arrival over time.')
        axs[2][1].legend(ncol=3, loc='best', shadow=True)
        axs[2][1].grid()

        ## TRAVEL TIME
        axs[2][2].axhline(y=TRAVEL_TIME_M, color='r', label='Baseline')
        axs[2][2].errorbar(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['ttime_mean'],
            yerr=self._aggregated_dataset[tag]['ttime_std'],
            capsize=5, label='Mean [std]', fmt='-o')
        axs[2][2].plot(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['ttime_median'], label='Median')
        axs[2][2].set(ylabel='Time [m]', title='Travel time over time.')
        axs[2][2].legend(ncol=3, loc='best', shadow=True)
        axs[2][2].grid()

        ## Mode: WAIT
        axs[3][0].axhline(0, linestyle=':')
        axs[3][0].errorbar(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['m_wait_mean'],
            yerr=self._aggregated_dataset[tag]['m_wait_std'],
            capsize=5, label='Mean [std]', fmt='-o')
        axs[3][0].plot(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['m_wait_median'], label='Median')
        axs[3][0].set(ylabel='People [#]', title='WAIT over time.')
        axs[3][0].legend(ncol=2, loc='best', shadow=True)
        axs[3][0].grid()

        ## Mode: CAR
        axs[3][1].axhline(0, linestyle=':')
        axs[3][1].axhline(200, linestyle=':', color='r', label='1/5')
        axs[3][1].errorbar(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['m_car_mean'],
            yerr=self._aggregated_dataset[tag]['m_car_std'],
            capsize=5, label='Mean [std]', fmt='-o')
        axs[3][1].plot(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['m_car_median'], label='Median')
        axs[3][1].set(ylabel='People [#]', title='CAR over time.')
        axs[3][1].legend(ncol=3, loc='best', shadow=True)
        axs[3][1].grid()

        ## Mode: PTW
        axs[3][2].axhline(0, linestyle=':')
        axs[3][2].axhline(200, linestyle=':', color='r', label='1/5')
        axs[3][2].errorbar(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['m_ptw_mean'],
            yerr=self._aggregated_dataset[tag]['m_ptw_std'],
            capsize=5, label='Mean [std]', fmt='-o')
        axs[3][2].plot(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['m_ptw_median'], label='Median')
        axs[3][2].set(ylabel='People [#]', title='PTW over time.')
        axs[3][2].legend(ncol=3, loc='best', shadow=True)
        axs[3][2].grid()

        ## Mode: WALK
        axs[4][0].axhline(0, linestyle=':')
        axs[4][0].axhline(200, linestyle=':', color='r', label='1/5')
        axs[4][0].errorbar(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['m_walk_mean'],
            yerr=self._aggregated_dataset[tag]['m_walk_std'],
            capsize=5, label='Mean [std]', fmt='-o')
        axs[4][0].plot(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['m_walk_median'], label='Median')
        axs[4][0].set(xlabel='Learning step', ylabel='People [#]', title='WALK over time.')
        axs[4][0].legend(ncol=3, loc='best', shadow=True)
        axs[4][0].grid()

        ## Mode: BICYCLE
        axs[4][1].axhline(0, linestyle=':')
        axs[4][1].axhline(200, linestyle=':', color='r', label='1/5')
        axs[4][1].errorbar(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['m_bicycle_mean'],
            yerr=self._aggregated_dataset[tag]['m_bicycle_std'],
            capsize=5, label='Mean [std]', fmt='-o')
        axs[4][1].plot(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['m_bicycle_median'], label='Median')
        axs[4][1].set(xlabel='Learning step', ylabel='People [#]', title='BICYCLE over time.')
        axs[4][1].legend(ncol=3, loc='best', shadow=True)
        axs[4][1].grid()

        ## Mode: PUBLIC TRANSPORTS
        axs[4][2].axhline(0, linestyle=':')
        axs[4][2].axhline(200, linestyle=':', color='r', label='1/5')
        axs[4][2].errorbar(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['m_pt_mean'],
            yerr=self._aggregated_dataset[tag]['m_pt_std'],
            capsize=5, label='Mean [std]', fmt='-o')
        axs[4][2].plot(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['m_pt_median'], label='Median')
        axs[4][2].set(
            xlabel='Learning step', ylabel='People [#]', title='PUBLIC TRANSPORTS over time.')
        axs[4][2].legend(ncol=3, loc='best', shadow=True)
        axs[4][2].grid()

        fig.savefig('{}/{}.overview_over_learning.svg'.format(self._output_dir, tag),
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
