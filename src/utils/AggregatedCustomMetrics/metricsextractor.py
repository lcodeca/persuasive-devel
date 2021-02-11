#!/usr/bin/env python3

""" Process the RLLIB metrics_XYZ.json """

import argparse
import collections
import cProfile
import io
import json
import os
import pstats
import re

from copy import deepcopy
from pprint import pformat

import numpy as np

from numpyencoder import NumpyEncoder
from tqdm import tqdm

def _argument_parser():
    """ Argument parser for the stats parser. """
    parser = argparse.ArgumentParser(
        description='RLLIB & SUMO Statistics parser.')
    parser.add_argument('--exp', required=True, type=str,
                        help='Experiment TAG.')
    parser.add_argument('--input-dir', required=True, type=str,
                        help='Input JSONs directory.')
    parser.add_argument('--max_metrics', type=int, default=10,
                        help='Maximum number of LATEST metrics to aggregate.')
    parser.add_argument('--output-dir', required=True, type=str,
                        help='Output aggregation JSON files.')
    parser.add_argument('--eval', dest='evaluation', action='store_true',
                        help='Set Evaluation vs Learning.')
    parser.set_defaults(evaluation=False)
    parser.add_argument('--reset', dest='reset', action='store_true',
                        help='Reset the metric if set.')
    parser.set_defaults(reset=False)
    parser.add_argument('--profiler', dest='profiler', action='store_true',
                        help='Enable cProfile.')
    parser.set_defaults(profiler=False)
    return parser.parse_args()

def _main():
    """ Process the RLLIB metrics/metrics_XYZ.json """

    config = _argument_parser()

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler = cProfile.Profile()
        profiler.enable()
    ## ========================              PROFILER              ======================== ##

    Extractor(
        config.exp, config.input_dir, config.max_metrics, config.evaluation,
        config.output_dir, config.reset).extract()

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler.disable()
        results = io.StringIO()
        pstats.Stats(profiler, stream=results).sort_stats('cumulative').print_stats(50)
        print('Profiler: \n{}'.format(pformat(results.getvalue())))
    ## ========================              PROFILER              ======================== ##

####################################################################################################

class Extractor():

    def __init__(self,
                 experiment: str, input_dir: str, max_metrics: int,
                 evaluation: bool, output_dir: str, reset: bool):
        self._exp = experiment
        self._eval = evaluation
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._max_metrics = max_metrics
        self._aggregated_dataset = None
        self._dataset_fname = {
            'reward': os.path.join(self._output_dir, 'reward.json'),
            'tot_reward': os.path.join(self._output_dir, 'tot_reward.json'),
            'arrival': os.path.join(self._output_dir, 'arrival.json'),
            'departure': os.path.join(self._output_dir, 'departure.json'),
            'missing': os.path.join(self._output_dir, 'missing.json'),
            'late': os.path.join(self._output_dir, 'late.json'),
            'waiting': os.path.join(self._output_dir, 'waiting.json'),
            'lateness': os.path.join(self._output_dir, 'lateness.json'),
            'travel_time': os.path.join(self._output_dir, 'travel_time.json'),
            'm_wait': os.path.join(self._output_dir, 'm_wait.json'),
            'm_car': os.path.join(self._output_dir, 'm_car.json'),
            'm_ptw': os.path.join(self._output_dir, 'm_ptw.json'),
            'm_walk': os.path.join(self._output_dir, 'm_walk.json'),
            'm_bicycle': os.path.join(self._output_dir, 'm_bicycle.json'),
            'm_public': os.path.join(self._output_dir, 'm_public.json'),
        }
        self._metric_to_mode = {
            'm_wait': 'wait',
            'm_car': 'passenger',
            'm_ptw': 'ptw',
            'm_walk': 'walk',
            'm_bicycle': 'bicycle',
            'm_public': 'public',
        }
        self._mode_to_action = {
            'wait': 0,
            'passenger': 1,
            'public': 2,
            'walk': 3,
            'bicycle': 4,
            'ptw': 5,
        }
        self._complete_metrics = None
        self._reset = {}
        for metric in self._dataset_fname:
            self._reset[metric] = reset
        print('Starting:', self._exp)

    def extract(self):
        self._extract()

    def _extract(self):
        files = self.alphanumeric_sort(os.listdir(self._input_dir))
        first_metric = len(files) - self._max_metrics
        if first_metric > 0:
            files = files[first_metric:]
        self._aggregate_metrics(files)

    def _aggregate_metrics(self, files):
        for filename in tqdm(files):
            with open(os.path.join(self._input_dir, filename), 'r') as jsonfile:
                self._complete_metrics = json.load(jsonfile)

                if 'action-to-mode' in self._complete_metrics['config']['env_config']['agent_init']:
                    self._mode_to_action = {'wait': 0,}
                    for _action, _mode in self._complete_metrics['config']['env_config']['agent_init']['action-to-mode'].items():
                        self._mode_to_action[_mode] = int(_action)
                    print('Using CONFIGURED action-to-mode:', self._mode_to_action)
                else:
                    print('Using default action-to-mode:', self._mode_to_action)

                if self._eval:
                    if 'evaluation' in self._complete_metrics:
                        self._complete_metrics['evaluation']['timesteps_total'] = \
                            self._complete_metrics['timesteps_total']
                        self._complete_metrics = self._complete_metrics['evaluation']
                    else:
                        raise Exception('Missing ["evaluation"] in file {}'.format(filename))

                self._extract_rewards()
                self._extract_tot_rewards()
                self._extract_overview()

    def _extract_rewards(self):
        self._load_aggregate_data('reward')
        current = self._complete_metrics['hist_stats']['policy_unique_reward']
        self._aggregated_dataset[self._exp].extend(deepcopy(current))
        self._save_aggregate_data('reward')

    def _extract_tot_rewards(self):
        self._load_aggregate_data('tot_reward')
        current = self._complete_metrics['hist_stats']['episode_reward']
        self._aggregated_dataset[self._exp].extend(deepcopy(current))
        self._save_aggregate_data('tot_reward')

    def _extract_overview(self):
        info_by_episode = self._complete_metrics['hist_stats']['info_by_agent']
        last_action_by_agent = self._complete_metrics['hist_stats']['last_action_by_agent']

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

        ## SAVE all the collected metrics

        ## MISSING
        self._load_aggregate_data('missing')
        self._aggregated_dataset[self._exp].extend(deepcopy(avg_missing))
        self._save_aggregate_data('missing')
        ## TOO LATE
        self._load_aggregate_data('late')
        self._aggregated_dataset[self._exp].extend(deepcopy(avg_too_late))
        self._save_aggregate_data('late')
        ## WAITING
        self._load_aggregate_data('waiting')
        self._aggregated_dataset[self._exp].extend(deepcopy(avg_waiting_s))
        self._save_aggregate_data('waiting')
        ## TOO LATE
        self._load_aggregate_data('lateness')
        self._aggregated_dataset[self._exp].extend(deepcopy(avg_lateness_s))
        self._save_aggregate_data('lateness')
        ## DEARTURE
        self._load_aggregate_data('departure')
        self._aggregated_dataset[self._exp].extend(deepcopy(avg_departure_s))
        self._save_aggregate_data('departure')
        ## TRAVEL TIME
        self._load_aggregate_data('travel_time')
        self._aggregated_dataset[self._exp].extend(deepcopy(avg_travel_time_s))
        self._save_aggregate_data('travel_time')
        ## ARRIVAL
        self._load_aggregate_data('arrival')
        self._aggregated_dataset[self._exp].extend(deepcopy(avg_arrival_s))
        self._save_aggregate_data('arrival')

        for _metric, _mode in self._metric_to_mode.items():
            self._load_aggregate_data(_metric)
            if _mode in self._mode_to_action:
                self._aggregated_dataset[self._exp].extend(
                    deepcopy(avg_modes[self._mode_to_action[_mode]]))
                # print(_mode, avg_modes[self._mode_to_action[_mode]])
            else:
                print('Missing:', _mode, self._exp)
            self._save_aggregate_data(_metric)

        # ## WAIT
        # self._load_aggregate_data('m_wait')
        # self._aggregated_dataset[self._exp].extend(deepcopy(avg_modes[0]))
        # self._save_aggregate_data('m_wait')
        # ## WALK
        # self._load_aggregate_data('m_walk')
        # self._aggregated_dataset[self._exp].extend(deepcopy(avg_modes[3]))
        # self._save_aggregate_data('m_walk')
        # ## BICYCLE
        # self._load_aggregate_data('m_bicycle')
        # self._aggregated_dataset[self._exp].extend(deepcopy(avg_modes[4]))
        # self._save_aggregate_data('m_bicycle')
        # ## PUBLIC TRANSPORTS
        # self._load_aggregate_data('m_public')
        # self._aggregated_dataset[self._exp].extend(deepcopy(avg_modes[2]))
        # self._save_aggregate_data('m_public')
        # ## CAR
        # self._load_aggregate_data('m_car')
        # self._aggregated_dataset[self._exp].extend(deepcopy(avg_modes[1]))
        # self._save_aggregate_data('m_car')
        # ## POWERED TWO-WHEELERS
        # self._load_aggregate_data('m_ptw')
        # self._aggregated_dataset[self._exp].extend(deepcopy(avg_modes[5]))
        # self._save_aggregate_data('m_ptw')

    def _load_aggregate_data(self, metric):
        if os.path.isfile(self._dataset_fname[metric]):
            with open(self._dataset_fname[metric], 'r') as jsonfile:
                self._aggregated_dataset = json.load(jsonfile)
        else:
            self._aggregated_dataset = dict()
        if self._reset[metric]:
            self._reset[metric] = False
            self._aggregated_dataset[self._exp] = []
            print('Resetting:', metric, self._exp)
        if self._exp not in self._aggregated_dataset:
            self._aggregated_dataset[self._exp] = []

    def _save_aggregate_data(self, metric):
        with open(self._dataset_fname[metric], 'w') as jsonfile:
            json.dump(self._aggregated_dataset, jsonfile, indent=2, cls=NumpyEncoder)

    @staticmethod
    def alphanumeric_sort(iterable):
        """
        Sorts the given iterable in the way that is expected.
        See: https://stackoverflow.com/questions/2669059/how-to-sort-alpha-numeric-set-in-python
        """
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(iterable, key=alphanum_key)



if __name__ == '__main__':
    _main()

####################################################################################################
