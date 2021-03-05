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

    GlobalTT(config.input_dir, config.output_dir).generate()
    logging.info('Done')

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler.disable()
        results = io.StringIO()
        pstats.Stats(profiler, stream=results).sort_stats('cumulative').print_stats(50)
        logging.info('Profiler: \n%s', pformat(results.getvalue()))
    ## ========================              PROFILER              ======================== ##

####################################################################################################

class GlobalTT(GenericGraphMaker):

    def __init__(self, input_dir, output_dir):
        _default = {
            'learning': {
                'timesteps_total': [],
                'global_tt_min': [],
                'global_tt_mean': [],
                'global_tt_std':[],
                'global_tt_median':[],
                'global_tt_max':[],
            },
            'evaluation': {
                'timesteps_total': [],
                'global_tt_min': [],
                'global_tt_mean': [],
                'global_tt_std':[],
                'global_tt_median':[],
                'global_tt_max':[],
            },
        }
        super().__init__(
            input_dir, output_dir,
            filename='global-travel-time.json',
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

                global_tt = []
                for episode in complete['hist_stats']['info_by_agent']:
                    _gtt = 0
                    for info in episode.values():
                        if not np.isnan(info['rtt']):
                            _gtt += info['rtt']/3600.0
                        # else:
                        #     print(info)
                    global_tt.append(_gtt)

                self._aggregated_dataset['learning']['global_tt_min'].append(
                    np.nanmin(global_tt))
                self._aggregated_dataset['learning']['global_tt_median'].append(
                    np.nanmedian(global_tt))
                self._aggregated_dataset['learning']['global_tt_mean'].append(
                    np.nanmean(global_tt))
                self._aggregated_dataset['learning']['global_tt_std'].append(
                    np.nanstd(global_tt))
                self._aggregated_dataset['learning']['global_tt_max'].append(
                    np.nanmax(global_tt))

                # EVALUATION
                if 'evaluation' in complete:
                    complete['evaluation']['timesteps_total'] = complete['timesteps_total']
                    complete = complete['evaluation']

                    global_tt = []
                    for episode in complete['hist_stats']['info_by_agent']:
                        _gtt = 0
                        for info in episode.values():
                            if not np.isnan(info['rtt']):
                                _gtt += info['rtt']/3600.0
                        global_tt.append(_gtt)

                    self._aggregated_dataset['evaluation']['timesteps_total'].append(
                        complete['timesteps_total'])
                    self._aggregated_dataset['evaluation']['global_tt_min'].append(
                        np.nanmin(global_tt))
                    self._aggregated_dataset['evaluation']['global_tt_median'].append(
                        np.nanmedian(global_tt))
                    self._aggregated_dataset['evaluation']['global_tt_mean'].append(
                        np.nanmean(global_tt))
                    self._aggregated_dataset['evaluation']['global_tt_std'].append(
                        np.nanstd(global_tt))
                    self._aggregated_dataset['evaluation']['global_tt_max'].append(
                        np.nanmax(global_tt))

    def _generate_graphs(self):
        # LEARNING
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.errorbar(
            self._aggregated_dataset['learning']['timesteps_total'],
            self._aggregated_dataset['learning']['global_tt_mean'],
            yerr=self._aggregated_dataset['learning']['global_tt_std'],
            capsize=5, label='Mean [std]', fmt='-o')
        ax.plot(
            self._aggregated_dataset['learning']['timesteps_total'],
            self._aggregated_dataset['learning']['global_tt_min'], label='Min')
        ax.plot(
            self._aggregated_dataset['learning']['timesteps_total'],
            self._aggregated_dataset['learning']['global_tt_max'], label='Max')
        ax.plot(
            self._aggregated_dataset['learning']['timesteps_total'],
            self._aggregated_dataset['learning']['global_tt_median'], label='Median')
        ax.set(
            xlabel='Learning step', ylabel='Time [h]',
            title='[L] Global Travel Time over time.')
        ax.legend(loc='best', ncol=4, shadow=True)
        ax.grid()
        fig.savefig('{}/learning.average_global_tt_over_learning.svg'.format(self._output_dir),
                    dpi=300, transparent=False, bbox_inches='tight')
        #plt.show()
        matplotlib.pyplot.close('all')

        # EVALUATION
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.errorbar(
            self._aggregated_dataset['evaluation']['timesteps_total'],
            self._aggregated_dataset['evaluation']['global_tt_mean'],
            yerr=self._aggregated_dataset['evaluation']['global_tt_std'],
            capsize=5, label='Mean [std]', fmt='-o')
        ax.plot(
            self._aggregated_dataset['evaluation']['timesteps_total'],
            self._aggregated_dataset['evaluation']['global_tt_min'], label='Min')
        ax.plot(
            self._aggregated_dataset['evaluation']['timesteps_total'],
            self._aggregated_dataset['evaluation']['global_tt_max'], label='Max')
        ax.plot(
            self._aggregated_dataset['evaluation']['timesteps_total'],
            self._aggregated_dataset['evaluation']['global_tt_median'], label='Median')
        ax.set(
            xlabel='Learning step', ylabel='Time [h]',
            title='[E] Global Travel Time over time.')
        ax.legend(loc='best', ncol=4, shadow=True)
        ax.grid()
        fig.savefig('{}/evaluation.average_global_tt_over_learning.svg'.format(self._output_dir),
                    dpi=300, transparent=False, bbox_inches='tight')
        #plt.show()
        matplotlib.pyplot.close('all')

####################################################################################################

if __name__ == '__main__':
    _main()

####################################################################################################
