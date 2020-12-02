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

    Performances(config.input_dir, config.output_dir).generate()
    logging.info('Done')

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler.disable()
        results = io.StringIO()
        pstats.Stats(profiler, stream=results).sort_stats('cumulative').print_stats(50)
        logging.info('Profiler: \n%s', pformat(results.getvalue()))
    ## ========================              PROFILER              ======================== ##

####################################################################################################

class Performances(GenericGraphMaker):

    def __init__(self, input_dir, output_dir):
        _default = {
            'timesteps_total': [],
            'sampler_perf': {
                'mean_env_wait_ms': [],
                'mean_raw_obs_processing_ms': [],
                'mean_inference_ms': [],
                'mean_action_processing_ms': [],
            },
            'timers': {
                'sample_time_ms':[],
                'sample_throughput': [],
                'load_time_ms': [],
                'load_throughput': [],
                'learn_time_ms': [],
                'learn_throughput': [],
                'update_time_ms': [],
            },
            'perf': {
                'cpu_util_percent': [],
                'ram_util_percent': [],
                'gpu_util_percent0': [],
                'vram_util_percent0': [],
            },
            'time_since_restore': [],
        }
        super().__init__(
            input_dir, output_dir,
            filename='sysperf.json',
            default=_default)

    def _find_last_metric(self):
        return len(self._aggregated_dataset['timesteps_total'])

    def _aggregate_metrics(self, files):
        for filename in tqdm(files):
            # print(filename)
            with open(os.path.join(self._input_dir, filename), 'r') as jsonfile:
                complete = json.load(jsonfile)

                self._aggregated_dataset['timesteps_total'].append(
                    complete['timesteps_total'])

                self._aggregated_dataset['sampler_perf']['mean_env_wait_ms'].append(
                    complete['sampler_perf']['mean_env_wait_ms'])
                self._aggregated_dataset['sampler_perf']['mean_raw_obs_processing_ms'].append(
                    complete['sampler_perf']['mean_raw_obs_processing_ms'])
                self._aggregated_dataset['sampler_perf']['mean_inference_ms'].append(
                    complete['sampler_perf']['mean_inference_ms'])
                self._aggregated_dataset['sampler_perf']['mean_action_processing_ms'].append(
                    complete['sampler_perf']['mean_action_processing_ms'])

                self._aggregated_dataset['timers']['sample_time_ms'].append(
                    complete['timers']['sample_time_ms'])
                self._aggregated_dataset['timers']['sample_throughput'].append(
                    complete['timers']['sample_throughput'])
                self._aggregated_dataset['timers']['load_time_ms'].append(
                    complete['timers']['load_time_ms'])
                self._aggregated_dataset['timers']['load_throughput'].append(
                    complete['timers']['load_throughput'])
                self._aggregated_dataset['timers']['learn_time_ms'].append(
                    complete['timers']['learn_time_ms'])
                self._aggregated_dataset['timers']['learn_throughput'].append(
                    complete['timers']['learn_throughput'])
                self._aggregated_dataset['timers']['update_time_ms'].append(
                    complete['timers']['update_time_ms'])

                self._aggregated_dataset['perf']['cpu_util_percent'].append(
                    complete['perf']['cpu_util_percent'])
                self._aggregated_dataset['perf']['ram_util_percent'].append(
                    complete['perf']['ram_util_percent'])
                self._aggregated_dataset['perf']['gpu_util_percent0'].append(
                    complete['perf']['gpu_util_percent0'])
                self._aggregated_dataset['perf']['vram_util_percent0'].append(
                    complete['perf']['vram_util_percent0'])

                self._aggregated_dataset['time_since_restore'].append(
                    complete['time_since_restore'])

    def _generate_graphs(self):

        fig, axs = plt.subplots(2, 2, sharex=True, figsize=(20, 15), constrained_layout=True)
        fig.suptitle('Sampler Performances')

        axs[0][0].plot(
            self._aggregated_dataset['timesteps_total'],
            self._aggregated_dataset['sampler_perf']['mean_env_wait_ms'],
            label='mean_env_wait')
            # color='blue', marker='o', linestyle='solid', linewidth=2, markersize=8)
        axs[0][0].set_ylabel('Time [ms]')
        axs[0][0].legend(loc='best', shadow=True)
        axs[0][0].grid(True)

        axs[0][1].plot(
            self._aggregated_dataset['timesteps_total'],
            self._aggregated_dataset['sampler_perf']['mean_raw_obs_processing_ms'],
            label='mean_raw_obs_processing')
            # color='blue', marker='o', linestyle='solid', linewidth=2, markersize=8)
        axs[0][1].set_ylabel('Time [ms]')
        axs[0][1].legend(loc='best', shadow=True)
        axs[0][1].grid(True)

        axs[1][0].plot(
            self._aggregated_dataset['timesteps_total'],
            self._aggregated_dataset['sampler_perf']['mean_inference_ms'],
            label='mean_inference_ms')
            # color='blue', marker='o', linestyle='solid', linewidth=2, markersize=8)
        axs[1][0].set_ylabel('Time [ms]')
        axs[1][0].set_xlabel('Learning Steps')
        axs[1][0].legend(loc='best', shadow=True)
        axs[1][0].grid(True)

        axs[1][1].plot(
            self._aggregated_dataset['timesteps_total'],
            self._aggregated_dataset['sampler_perf']['mean_action_processing_ms'],
            label='mean_action_processing')
            # color='blue', marker='o', linestyle='solid', linewidth=2, markersize=8)
        axs[1][1].set_ylabel('Time [ms]')
        axs[1][1].set_xlabel('Learning Steps')
        axs[1][1].legend(loc='best', shadow=True)
        axs[1][1].grid(True)

        fig.savefig('{}/sysperf.sampler_perf_over_learning.svg'.format(self._output_dir),
                    dpi=300, transparent=False, bbox_inches='tight')
        #plt.show()
        matplotlib.pyplot.close('all')

        ########################################################################

        fig, axs = plt.subplots(4, 2, sharex=True, figsize=(20, 20), constrained_layout=True)
        fig.suptitle('Timers')

        axs[0][0].plot(
            self._aggregated_dataset['timesteps_total'],
            self._aggregated_dataset['timers']['sample_time_ms'], label='Sample')
            # color='blue', marker='o', linestyle='solid', linewidth=2, markersize=8)
        axs[0][0].set_ylabel('Time [ms]')
        axs[0][0].legend(loc='best', shadow=True)
        axs[0][0].grid(True)

        axs[0][1].plot(
            self._aggregated_dataset['timesteps_total'],
            self._aggregated_dataset['timers']['sample_throughput'], label='Sample')
            # color='blue', marker='o', linestyle='solid', linewidth=2, markersize=8)
        axs[0][1].set_ylabel('Throughput')
        axs[0][1].legend(loc='best', shadow=True)
        axs[0][1].grid(True)

        axs[1][0].plot(
            self._aggregated_dataset['timesteps_total'],
            self._aggregated_dataset['timers']['load_time_ms'], label='Load')
            # color='blue', marker='o', linestyle='solid', linewidth=2, markersize=8)
        axs[1][0].set_ylabel('Time [ms]')
        axs[1][0].legend(loc='best', shadow=True)
        axs[1][0].grid(True)

        axs[1][1].plot(
            self._aggregated_dataset['timesteps_total'],
            self._aggregated_dataset['timers']['load_throughput'], label='Load')
            # color='blue', marker='o', linestyle='solid', linewidth=2, markersize=8)
        axs[1][1].set_ylabel('Throughput')
        axs[1][1].legend(loc='best', shadow=True)
        axs[1][1].grid(True)

        axs[2][0].plot(
            self._aggregated_dataset['timesteps_total'],
            self._aggregated_dataset['timers']['learn_time_ms'], label='Learn')
            # color='blue', marker='o', linestyle='solid', linewidth=2, markersize=8)
        axs[2][0].set_ylabel('Time [ms]')
        axs[2][0].legend(loc='best', shadow=True)
        axs[2][0].grid(True)

        axs[2][1].plot(
            self._aggregated_dataset['timesteps_total'],
            self._aggregated_dataset['timers']['learn_throughput'], label='Learn')
            # color='blue', marker='o', linestyle='solid', linewidth=2, markersize=8)
        axs[2][1].set_ylabel('Throughput')
        axs[2][1].legend(loc='best', shadow=True)
        axs[2][1].grid(True)

        axs[3][0].plot(
            self._aggregated_dataset['timesteps_total'],
            self._aggregated_dataset['timers']['update_time_ms'], label='Update')
            # color='blue', marker='o', linestyle='solid', linewidth=2, markersize=8)
        axs[3][0].set_ylabel('Time [ms]')
        axs[3][0].set_xlabel('Learning Steps')
        axs[3][0].legend(loc='best', shadow=True)
        axs[3][0].grid(True)

        axs[3][1].set_xlabel('Learning Steps')
        axs[3][1].set_ylabel('Throughput')
        axs[3][1].grid(True)

        fig.savefig('{}/sysperf.timers_over_learning.svg'.format(self._output_dir),
                    dpi=300, transparent=False, bbox_inches='tight')
        #plt.show()
        matplotlib.pyplot.close('all')

        ########################################################################

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.plot(
            self._aggregated_dataset['timesteps_total'],
            self._aggregated_dataset['perf']['cpu_util_percent'],
            label="CPU")
        ax.plot(
            self._aggregated_dataset['timesteps_total'],
            self._aggregated_dataset['perf']['ram_util_percent'],
            label="RAM")
        ax.plot(
            self._aggregated_dataset['timesteps_total'],
            self._aggregated_dataset['perf']['gpu_util_percent0'],
            label="GPU0")
        ax.plot(
            self._aggregated_dataset['timesteps_total'],
            self._aggregated_dataset['perf']['vram_util_percent0'],
            label="VRAM")
        ax.set(xlabel='Learning step', ylabel='Load [%]', title='System Performances over Learning')
        ax.legend(loc='best', ncol=4, shadow=True)
        ax.grid()
        fig.savefig('{}/sysperf.perf_over_learning.svg'.format(self._output_dir),
                    dpi=300, transparent=False, bbox_inches='tight')
        #plt.show()
        matplotlib.pyplot.close('all')

        ########################################################################

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.plot(
            self._aggregated_dataset['timesteps_total'],
            self._aggregated_dataset['time_since_restore'])
        ax.set(xlabel='Learning step', ylabel='Time [s]', title='Time Since Restore over Learning')
        # ax.legend(loc='best', ncol=4, shadow=True)
        ax.grid()
        fig.savefig('{}/sysperf.time_restore_over_learning.svg'.format(self._output_dir),
                    dpi=300, transparent=False, bbox_inches='tight')
        #plt.show()
        matplotlib.pyplot.close('all')

####################################################################################################

    # def perf_over_timesteps_total(self):
    #     logger.info('Computing the system performances over the timesteps total.')
    #     perfs = collections.defaultdict(list)
    #     with open(self.input, 'r') as jsonfile:
    #         for row in tqdm(jsonfile): # enumerate cannot be used due to the size of the file
    #             complete = json.loads(row)
    #             for metric, value in complete['perf'].items():
    #                 perfs[metric].append((complete['timesteps_total'], value))

    #     fig, ax = plt.subplots(figsize=(15, 10))
    #     for metric, values in perfs.items():
    #         xval = []
    #         yval = []
    #         for timestep, val in values:
    #             xval.append(timestep)
    #             yval.append(val)
    #         ax.plot(xval, yval, label=metric)
    #     ax.set(xlabel='Timesteps', ylabel='Load [%]', title='System Performances')
    #     ax.legend(loc='best', shadow=True)
    #     ax.grid()
    #     fig.savefig('{}.sysperf_over_learning.svg'.format(self.prefix),
    #                 dpi=300, transparent=False, bbox_inches='tight')
    #     #plt.show()
    #     matplotlib.pyplot.close('all')

####################################################################################################

if __name__ == '__main__':
    _main()

####################################################################################################
