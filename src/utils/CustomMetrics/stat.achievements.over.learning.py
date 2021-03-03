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
        _default = {
            'learning': {
                'timesteps_total': [],
                '1st_choice': [],
                '1st_choice_num_late': [],
                '1st_choice_lateness': [],
                '1st_choice_waiting': [],
                '2nd_choice': [],
                '2nd_choice_num_late': [],
                '2nd_choice_lateness': [],
                '2nd_choice_waiting': [],
                '3rd_choice': [],
                '3rd_choice_num_late': [],
                '3rd_choice_lateness': [],
                '3rd_choice_waiting': [],
                'rest_choice': [],
                'rest_choice_num_late': [],
                'rest_choice_lateness': [],
                'rest_choice_waiting': [],
            },
            'evaluation': {
                'timesteps_total': [],
                '1st_choice': [],
                '1st_choice_num_late': [],
                '1st_choice_lateness': [],
                '1st_choice_waiting': [],
                '2nd_choice': [],
                '2nd_choice_num_late': [],
                '2nd_choice_lateness': [],
                '2nd_choice_waiting': [],
                '3rd_choice': [],
                '3rd_choice_num_late': [],
                '3rd_choice_lateness': [],
                '3rd_choice_waiting': [],
                'rest_choice': [],
                'rest_choice_num_late': [],
                'rest_choice_lateness': [],
                'rest_choice_waiting': [],
            },
        }
        super().__init__(
            input_dir, output_dir,
            filename='achievements.json',
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

                _1st_choice = []
                _1st_choice_num_late = []
                _1st_choice_lateness = []
                _1st_choice_waiting = []
                _2nd_choice = []
                _2nd_choice_num_late = []
                _2nd_choice_lateness = []
                _2nd_choice_waiting = []
                _3rd_choice = []
                _3rd_choice_num_late = []
                _3rd_choice_lateness = []
                _3rd_choice_waiting = []
                _rest_choice = []
                _rest_choice_num_late = []
                _rest_choice_lateness = []
                _rest_choice_waiting = []

                for episode in complete['hist_stats']['info_by_agent']:
                    _curr_1st_choice = 0
                    _curr_1st_choice_num_late = 0
                    _curr_1st_choice_lateness = []
                    _curr_1st_choice_waiting = []
                    _curr_2nd_choice = 0
                    _curr_2nd_choice_num_late = 0
                    _curr_2nd_choice_lateness = []
                    _curr_2nd_choice_waiting = []
                    _curr_3rd_choice = 0
                    _curr_3rd_choice_num_late = 0
                    _curr_3rd_choice_lateness = []
                    _curr_3rd_choice_waiting = []
                    _curr_rest_choice = 0
                    _curr_rest_choice_num_late = 0
                    _curr_rest_choice_lateness = []
                    _curr_rest_choice_waiting = []

                    for info in episode.values():
                        if 'preferences' not in info['init']:
                            continue
                        pref = sorted([(p, m) for m, p in info['init']['preferences'].items()])
                        if not np.isnan(info['arrival']):
                            # A choice was made.
                            if info['mode'] == pref[0][1]:
                                _curr_1st_choice += 1
                                if np.isnan(info['wait']):
                                    # late
                                    _curr_1st_choice_num_late += 1
                                    _curr_1st_choice_lateness.append(
                                        (info['arrival']-info['init']['exp-arrival'])/60.0)
                                else:
                                    _curr_1st_choice_waiting.append(
                                        info['wait']/60.0)
                            elif info['mode'] == pref[1][1]:
                                _curr_2nd_choice += 1
                                if np.isnan(info['wait']):
                                    # late
                                    _curr_2nd_choice_num_late += 1
                                    _curr_2nd_choice_lateness.append(
                                        (info['arrival']-info['init']['exp-arrival'])/60.0)
                                else:
                                    _curr_2nd_choice_waiting.append(
                                        info['wait']/60.0)
                            elif info['mode'] == pref[2][1]:
                                _curr_3rd_choice += 1
                                if np.isnan(info['wait']):
                                    # late
                                    _curr_3rd_choice_num_late += 1
                                    _curr_3rd_choice_lateness.append(
                                        (info['arrival']-info['init']['exp-arrival'])/60.0)
                                else:
                                    _curr_3rd_choice_waiting.append(
                                        info['wait']/60.0)
                            else:
                                _curr_rest_choice += 1
                                if np.isnan(info['wait']):
                                    # late
                                    _curr_rest_choice_num_late += 1
                                    _curr_rest_choice_lateness.append(
                                        (info['arrival']-info['init']['exp-arrival'])/60.0)
                                else:
                                    _curr_rest_choice_waiting.append(
                                        info['wait']/60.0)

                    _1st_choice.append(_curr_1st_choice)
                    _1st_choice_num_late.append(_curr_1st_choice_num_late)
                    _1st_choice_lateness.append(np.mean(_curr_1st_choice_lateness))
                    _1st_choice_waiting.append(np.mean(_curr_1st_choice_waiting))
                    _2nd_choice.append(_curr_2nd_choice)
                    _2nd_choice_num_late.append(_curr_2nd_choice_num_late)
                    _2nd_choice_lateness.append(np.mean(_curr_2nd_choice_lateness))
                    _2nd_choice_waiting.append(np.mean(_curr_2nd_choice_waiting))
                    _3rd_choice.append(_curr_3rd_choice)
                    _3rd_choice_num_late.append(_curr_3rd_choice_num_late)
                    _3rd_choice_lateness.append(np.mean(_curr_3rd_choice_lateness))
                    _3rd_choice_waiting.append(np.mean(_curr_3rd_choice_waiting))
                    _rest_choice.append(_curr_rest_choice)
                    _rest_choice_num_late.append(_curr_rest_choice_num_late)
                    _rest_choice_lateness.append(np.mean(_curr_rest_choice_lateness))
                    _rest_choice_waiting.append(np.mean(_curr_rest_choice_waiting))

                self._aggregated_dataset['learning']['1st_choice'].append(
                    np.nanmean(_1st_choice))
                self._aggregated_dataset['learning']['1st_choice_num_late'].append(
                    np.nanmean(_1st_choice_num_late))
                self._aggregated_dataset['learning']['1st_choice_lateness'].append(
                    np.nanmean(_1st_choice_lateness))
                self._aggregated_dataset['learning']['1st_choice_waiting'].append(
                    np.nanmean(_1st_choice_waiting))

                self._aggregated_dataset['learning']['2nd_choice'].append(
                    np.nanmean(_2nd_choice))
                self._aggregated_dataset['learning']['2nd_choice_num_late'].append(
                    np.nanmean(_2nd_choice_num_late))
                self._aggregated_dataset['learning']['2nd_choice_lateness'].append(
                    np.nanmean(_2nd_choice_lateness))
                self._aggregated_dataset['learning']['2nd_choice_waiting'].append(
                    np.nanmean(_2nd_choice_waiting))

                self._aggregated_dataset['learning']['3rd_choice'].append(
                    np.nanmean(_3rd_choice))
                self._aggregated_dataset['learning']['3rd_choice_num_late'].append(
                    np.nanmean(_3rd_choice_num_late))
                self._aggregated_dataset['learning']['3rd_choice_lateness'].append(
                    np.nanmean(_3rd_choice_lateness))
                self._aggregated_dataset['learning']['3rd_choice_waiting'].append(
                    np.nanmean(_3rd_choice_waiting))

                self._aggregated_dataset['learning']['rest_choice'].append(
                    np.nanmean(_rest_choice))
                self._aggregated_dataset['learning']['rest_choice_num_late'].append(
                    np.nanmean(_rest_choice_num_late))
                self._aggregated_dataset['learning']['rest_choice_lateness'].append(
                    np.nanmean(_rest_choice_lateness))
                self._aggregated_dataset['learning']['rest_choice_waiting'].append(
                    np.nanmean(_rest_choice_waiting))

                # EVALUATION
                if 'evaluation' in complete:
                    complete['evaluation']['timesteps_total'] = complete['timesteps_total']
                    complete = complete['evaluation']

                    self._aggregated_dataset['evaluation']['timesteps_total'].append(
                        complete['timesteps_total'])

                    _1st_choice = []
                    _1st_choice_num_late = []
                    _1st_choice_lateness = []
                    _1st_choice_waiting = []
                    _2nd_choice = []
                    _2nd_choice_num_late = []
                    _2nd_choice_lateness = []
                    _2nd_choice_waiting = []
                    _3rd_choice = []
                    _3rd_choice_num_late = []
                    _3rd_choice_lateness = []
                    _3rd_choice_waiting = []
                    _rest_choice = []
                    _rest_choice_num_late = []
                    _rest_choice_lateness = []
                    _rest_choice_waiting = []

                    for episode in complete['hist_stats']['info_by_agent']:
                        _curr_1st_choice = 0
                        _curr_1st_choice_num_late = 0
                        _curr_1st_choice_lateness = []
                        _curr_1st_choice_waiting = []
                        _curr_2nd_choice = 0
                        _curr_2nd_choice_num_late = 0
                        _curr_2nd_choice_lateness = []
                        _curr_2nd_choice_waiting = []
                        _curr_3rd_choice = 0
                        _curr_3rd_choice_num_late = 0
                        _curr_3rd_choice_lateness = []
                        _curr_3rd_choice_waiting = []
                        _curr_rest_choice = 0
                        _curr_rest_choice_num_late = 0
                        _curr_rest_choice_lateness = []
                        _curr_rest_choice_waiting = []

                        for info in episode.values():
                            if 'preferences' not in info['init']:
                                continue
                            pref = sorted([(p, m) for m, p in info['init']['preferences'].items()])
                            if not np.isnan(info['arrival']):
                                # A choice was made.
                                if info['mode'] == pref[0][1]:
                                    _curr_1st_choice += 1
                                    if np.isnan(info['wait']):
                                        # late
                                        _curr_1st_choice_num_late += 1
                                        _curr_1st_choice_lateness.append(
                                            (info['arrival']-info['init']['exp-arrival'])/60.0)
                                    else:
                                        _curr_1st_choice_waiting.append(
                                            info['wait']/60.0)
                                elif info['mode'] == pref[1][1]:
                                    _curr_2nd_choice += 1
                                    if np.isnan(info['wait']):
                                        # late
                                        _curr_2nd_choice_num_late += 1
                                        _curr_2nd_choice_lateness.append(
                                            (info['arrival']-info['init']['exp-arrival'])/60.0)
                                    else:
                                        _curr_2nd_choice_waiting.append(
                                            info['wait']/60.0)
                                elif info['mode'] == pref[2][1]:
                                    _curr_3rd_choice += 1
                                    if np.isnan(info['wait']):
                                        # late
                                        _curr_3rd_choice_num_late += 1
                                        _curr_3rd_choice_lateness.append(
                                            (info['arrival']-info['init']['exp-arrival'])/60.0)
                                    else:
                                        _curr_3rd_choice_waiting.append(
                                            info['wait']/60.0)
                                else:
                                    _curr_rest_choice += 1
                                    if np.isnan(info['wait']):
                                        # late
                                        _curr_rest_choice_num_late += 1
                                        _curr_rest_choice_lateness.append(
                                            (info['arrival']-info['init']['exp-arrival'])/60.0)
                                    else:
                                        _curr_rest_choice_waiting.append(
                                            info['wait']/60.0)

                        _1st_choice.append(_curr_1st_choice)
                        _1st_choice_num_late.append(_curr_1st_choice_num_late)
                        _1st_choice_lateness.append(np.mean(_curr_1st_choice_lateness))
                        _1st_choice_waiting.append(np.mean(_curr_1st_choice_waiting))
                        _2nd_choice.append(_curr_2nd_choice)
                        _2nd_choice_num_late.append(_curr_2nd_choice_num_late)
                        _2nd_choice_lateness.append(np.mean(_curr_2nd_choice_lateness))
                        _2nd_choice_waiting.append(np.mean(_curr_2nd_choice_waiting))
                        _3rd_choice.append(_curr_3rd_choice)
                        _3rd_choice_num_late.append(_curr_3rd_choice_num_late)
                        _3rd_choice_lateness.append(np.mean(_curr_3rd_choice_lateness))
                        _3rd_choice_waiting.append(np.mean(_curr_3rd_choice_waiting))
                        _rest_choice.append(_curr_rest_choice)
                        _rest_choice_num_late.append(_curr_rest_choice_num_late)
                        _rest_choice_lateness.append(np.mean(_curr_rest_choice_lateness))
                        _rest_choice_waiting.append(np.mean(_curr_rest_choice_waiting))

                    self._aggregated_dataset['evaluation']['1st_choice'].append(
                        np.nanmean(_1st_choice))
                    self._aggregated_dataset['evaluation']['1st_choice_num_late'].append(
                        np.nanmean(_1st_choice_num_late))
                    self._aggregated_dataset['evaluation']['1st_choice_lateness'].append(
                        np.nanmean(_1st_choice_lateness))
                    self._aggregated_dataset['evaluation']['1st_choice_waiting'].append(
                        np.nanmean(_1st_choice_waiting))

                    self._aggregated_dataset['evaluation']['2nd_choice'].append(
                        np.nanmean(_2nd_choice))
                    self._aggregated_dataset['evaluation']['2nd_choice_num_late'].append(
                        np.nanmean(_2nd_choice_num_late))
                    self._aggregated_dataset['evaluation']['2nd_choice_lateness'].append(
                        np.nanmean(_2nd_choice_lateness))
                    self._aggregated_dataset['evaluation']['2nd_choice_waiting'].append(
                        np.nanmean(_2nd_choice_waiting))

                    self._aggregated_dataset['evaluation']['3rd_choice'].append(
                        np.nanmean(_3rd_choice))
                    self._aggregated_dataset['evaluation']['3rd_choice_num_late'].append(
                        np.nanmean(_3rd_choice_num_late))
                    self._aggregated_dataset['evaluation']['3rd_choice_lateness'].append(
                        np.nanmean(_3rd_choice_lateness))
                    self._aggregated_dataset['evaluation']['3rd_choice_waiting'].append(
                        np.nanmean(_3rd_choice_waiting))

                    self._aggregated_dataset['evaluation']['rest_choice'].append(
                        np.nanmean(_rest_choice))
                    self._aggregated_dataset['evaluation']['rest_choice_num_late'].append(
                        np.nanmean(_rest_choice_num_late))
                    self._aggregated_dataset['evaluation']['rest_choice_lateness'].append(
                        np.nanmean(_rest_choice_lateness))
                    self._aggregated_dataset['evaluation']['rest_choice_waiting'].append(
                        np.nanmean(_rest_choice_waiting))

        # pprint(self._aggregated_dataset)
        # sys.exit()

    def _generate_graphs_helper(self, tag, lbl):
        fig, axs = plt.subplots(4, 4, figsize=(45, 25),
                                constrained_layout=True, sharey='row') # sharex='col',
        fig.suptitle('{} Aggregated Achievements over Learning'.format(lbl))

        ## 1ST Choice
        axs[0][0].axhline(y=0, linestyle=':')
        axs[0][0].plot(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['1st_choice'], label='Mean')
        axs[0][0].set(ylabel='Agents [#]', title='1ST Choice')
        # axs[0][0].legend(ncol=3, loc='best', shadow=True)
        axs[0][0].grid()
        ## ## ##
        axs[1][0].axhline(y=0, linestyle=':')
        axs[1][0].plot(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['1st_choice_num_late'], label='Mean')
        axs[1][0].set(ylabel='Late Agents [#]')
        # axs[0][0].legend(ncol=3, loc='best', shadow=True)
        axs[1][0].grid()
        ## ## ##
        axs[2][0].axhline(y=0, linestyle=':')
        axs[2][0].plot(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['1st_choice_lateness'], label='Mean')
        axs[2][0].set(ylabel='Lateness [min]')
        # axs[0][0].legend(ncol=3, loc='best', shadow=True)
        axs[2][0].grid()
        ## ## ##
        axs[3][0].axhline(y=0, linestyle=':')
        axs[3][0].plot(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['1st_choice_waiting'], label='Mean')
        axs[3][0].set(xlabel='Learning step', ylabel='Waiting [min]')
        # axs[0][0].legend(ncol=3, loc='best', shadow=True)
        axs[3][0].grid()

        ## 2ND Choice
        axs[0][1].axhline(y=0, linestyle=':')
        axs[0][1].plot(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['2nd_choice'], label='Mean')
        axs[0][1].set(ylabel='Agents [#]', title='2ND Choice')
        # axs[0][0].legend(ncol=3, loc='best', shadow=True)
        axs[0][1].grid()
        ## ## ##
        axs[1][1].axhline(y=0, linestyle=':')
        axs[1][1].plot(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['2nd_choice_num_late'], label='Mean')
        axs[1][1].set(ylabel='Late Agents [#]')
        # axs[0][0].legend(ncol=3, loc='best', shadow=True)
        axs[1][1].grid()
        ## ## ##
        axs[2][1].axhline(y=0, linestyle=':')
        axs[2][1].plot(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['2nd_choice_lateness'], label='Mean')
        axs[2][1].set(ylabel='Lateness [min]')
        # axs[0][0].legend(ncol=3, loc='best', shadow=True)
        axs[2][1].grid()
        ## ## ##
        axs[3][1].axhline(y=0, linestyle=':')
        axs[3][1].plot(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['2nd_choice_waiting'], label='Mean')
        axs[3][1].set(xlabel='Learning step', ylabel='Waiting [min]')
        # axs[0][0].legend(ncol=3, loc='best', shadow=True)
        axs[3][1].grid()

        ## 3RD Choice
        axs[0][2].axhline(y=0, linestyle=':')
        axs[0][2].plot(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['3rd_choice'], label='Mean')
        axs[0][2].set(ylabel='Agents [#]', title='3RD Choice')
        # axs[0][2].legend(ncol=3, loc='best', shadow=True)
        axs[0][2].grid()
        ## ## ##
        axs[1][2].axhline(y=0, linestyle=':')
        axs[1][2].plot(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['3rd_choice_num_late'], label='Mean')
        axs[1][2].set(ylabel='Late Agents [#]')
        # axs[0][0].legend(ncol=3, loc='best', shadow=True)
        axs[1][2].grid()
        ## ## ##
        axs[2][2].axhline(y=0, linestyle=':')
        axs[2][2].plot(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['3rd_choice_lateness'], label='Mean')
        axs[2][2].set(ylabel='Lateness [min]')
        # axs[0][0].legend(ncol=3, loc='best', shadow=True)
        axs[2][2].grid()
        ## ## ##
        axs[3][2].axhline(y=0, linestyle=':')
        axs[3][2].plot(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['3rd_choice_waiting'], label='Mean')
        axs[3][2].set(xlabel='Learning step', ylabel='Waiting [min]')
        # axs[0][0].legend(ncol=3, loc='best', shadow=True)
        axs[3][2].grid()

        ## REST Choice
        axs[0][3].axhline(y=0, linestyle=':')
        axs[0][3].plot(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['rest_choice'], label='Mean')
        axs[0][3].set(ylabel='Agents [#]', title='4+ Choice')
        # axs[0][3].legend(ncol=3, loc='best', shadow=True)
        axs[0][3].grid()
        ## ## ##
        axs[1][3].axhline(y=0, linestyle=':')
        axs[1][3].plot(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['rest_choice_num_late'], label='Mean')
        axs[1][3].set(ylabel='Late Agents [#]')
        # axs[0][0].legend(ncol=3, loc='best', shadow=True)
        axs[1][3].grid()
        ## ## ##
        axs[2][3].axhline(y=0, linestyle=':')
        axs[2][3].plot(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['rest_choice_lateness'], label='Mean')
        axs[2][3].set(ylabel='Lateness [min]')
        # axs[0][0].legend(ncol=3, loc='best', shadow=True)
        axs[2][3].grid()
        ## ## ##
        axs[3][3].axhline(y=0, linestyle=':')
        axs[3][3].plot(
            self._aggregated_dataset[tag]['timesteps_total'],
            self._aggregated_dataset[tag]['rest_choice_waiting'], label='Mean')
        axs[3][3].set(xlabel='Learning step', ylabel='Waiting [min]')
        # axs[0][0].legend(ncol=3, loc='best', shadow=True)
        axs[3][3].grid()

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
