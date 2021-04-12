#!/usr/bin/env python3

""" Process the JSON metrics in R-friendly CSV. """

import argparse
import collections
import cProfile
import csv
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
    parser.add_argument('--input', required=True, type=str,
                        help='Input JSON file.')
    parser.add_argument('--output', required=True, type=str,
                        help='Output CSV file.')
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

    JSON2CSV(config.input, config.output).process()

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler.disable()
        results = io.StringIO()
        pstats.Stats(profiler, stream=results).sort_stats('cumulative').print_stats(50)
        print('Profiler: \n{}'.format(pformat(results.getvalue())))
    ## ========================              PROFILER              ======================== ##

####################################################################################################

class JSON2CSV():

    _REWARDS = {
        'wSimplifiedReward': 'Simple',
        'wSimpleTTReward': 'RTT',
        'wSimpleTTCoopReward': 'RTTCoop'
    }

    _START_DISTR = {
        '30_2_30_3_45_4_45_4_60_5_60_5_60_5_60_5': 'All',
        '30_2_w': '30_2',
        '30_3_w': '30_3',
        '30_3_G': '30_3',
        '45_4_w': '45_4',
        '60_5_w': '60_5',
    }

    _DEFAULT = ['Name',
                '30_2 Mean [m]', '30_2 StDev [s]', '30_3 Mean [m]', '30_3 StDev [s]',
                '45_4 Mean [m]', '45_4 StDev [s]', '60_5 Mean [m]', '60_5 StDev [s]',
                'All Mean [m]', 'All StDev [s]',]


    def __init__(self, in_file: str, out_file: str):
        self._input = in_file
        self._output = out_file
        self._json = None
        self._to_csv = dict()

    def process(self):
        self._process()
        self._save_to_csv()

    def _get_reward_model(self, exp):
        for model, tag in self._REWARDS.items():
            if model in exp:
                return tag
        raise Exception('REWARD MODEL ERROR for {}'.format(exp))

    def _get_reward_model(self, exp):
        for model, tag in self._REWARDS.items():
            if model in exp:
                return tag
        raise Exception('REWARD MODEL ERROR for {}'.format(exp))

    def _get_start_distr(self, exp):
        for distr, tag in self._START_DISTR.items():
            if distr in exp:
                return tag
        raise Exception('START DISTRIB ERROR for {}'.format(exp))

    def _get_background_traffic(self, exp):
        if 'noBGTraffic' in exp:
            return False
        return True

    def _get_experiment(self, exp):
        tail = exp.split('_')[-1]
        if tail == '128':
            # base experiment
            tail = ''
        if 'wOwnership' in exp:
            return 'Own {}'.format(tail).strip()
        if 'wPreferences' in exp:
            return 'Pref {}'.format(tail).strip()
        if tail:
            return tail
        return 'Base'

    def _get_experiment_lbl(self, exp):
        reward = self._get_reward_model(exp)
        # start_distr = self._get_start_distr(exp)
        bgt = self._get_background_traffic(exp)
        if bgt:
            bgt = 'wBGT'
        else:
            bgt = ''
        tail = exp.split('_')[-1]
        if tail == '128':
            # base experiment
            tail = ''
        if 'wOwnership' in exp:
            return '{} Own {} {}'.format(reward, tail, bgt).strip()
        if 'wPreferences' in exp:
            return '{} Pref {} {}'.format(reward, tail, bgt).strip()
        return '{} {} {}'.format(reward, tail, bgt).strip()

    def _process(self):
        self._json = json.load(open(self._input))
        for experiment, values in tqdm(self._json.items()):
            _reward = self._get_reward_model(experiment)
            _start_distr = self._get_start_distr(experiment)
            _bgt = self._get_background_traffic(experiment)
            _exp = self._get_experiment(experiment)
            _lbl = self._get_experiment_lbl(experiment)
            # print('-{}-'.format(_exp))
            mean = round(np.nanmean(values)/60, 2)
            stdev = round(np.nanstd(values), 2)
            current = None
            if _lbl not in self._to_csv:
                current = {
                    # 'Reward Model': _reward,
                    # 'Backgound Traffic': _bgt,
                    # 'Experiment': _exp,
                    'Name': _lbl,
                    '{} Mean [m]'.format(_start_distr): mean,
                    '{} StDev [s]'.format(_start_distr): stdev,
                }
                self._to_csv[_lbl] = current
            else:
                self._to_csv[_lbl]['{} Mean [m]'.format(_start_distr)] = mean
                self._to_csv[_lbl]['{} StDev [s]'.format(_start_distr)] = stdev
            # for item in self._to_csv[_lbl]:
            #     self._DEFAULT.add(item)

    def _save_to_csv(self):
        with open(self._output, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self._DEFAULT)
            writer.writeheader()
            for data in self._to_csv.values():
                writer.writerow(data)

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
