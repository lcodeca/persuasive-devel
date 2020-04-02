#!/usr/bin/env python3

""" Process the RLLIB logs/result.json """

import argparse
import collections
import cProfile
import io
import json
import logging
import os
from pprint import pformat, pprint
import pstats
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

LOGGER = logging.getLogger(__name__)

def _argument_parser():
    """ Argument parser for the stats parser. """
    parser = argparse.ArgumentParser(
        description='RLLIB & SUMO Statistics parser.')
    parser.add_argument(
        '--dir', required=True, type=str, 
        help='Main directory.')
    parser.add_argument(
        '--pattern', required=True, type=str, 
        help='Input substring to look for the JSON file in multiple folders.')
    parser.add_argument(
        '--prefix', default='stats', help='Output prefix for the processed data.')
    parser.add_argument(
        '--profiler', dest='profiler', action='store_true', help='Enable cProfile.')
    parser.set_defaults(profiler=False)
    return parser.parse_args()

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

class Statistics(object):
    """ Loads the result.json file as a time series. """

    def __init__(self, config):
        self.config = config

    def reward_over_timesteps_total(self):
        LOGGER.info('Loading %s..', self.config.input)
        x_coords = []
        y_coords = []
        median_y = []
        min_y = []
        max_y = []
        std_y = []
        with open(self.config.input, 'r') as jsonfile:
            for row in jsonfile: # enumerate cannot be used due to the size of the file
                complete = json.loads(row)
                x_coords.append(complete['timesteps_total'])
                y_coords.append(complete['episode_reward_mean'])
                _rewards = []
                for policy in complete['policies'].values():
                    _rewards.append(policy['stats']['agent_reward'])
                min_y.append(min(_rewards))
                max_y.append(max(_rewards))
                median_y.append(np.median(_rewards))
                std_y.append(np.std(_rewards))

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.errorbar(x_coords, y_coords, yerr=std_y, capsize=5, label='Mean [std]', fmt='-o')
        ax.plot(x_coords, min_y, label='Min')
        ax.plot(x_coords, max_y, label='Max')
        ax.plot(x_coords, median_y, label='Median')
        ax.set(xlabel='Learning step', ylabel='Reward',
            title='Reward over time')
        ax.legend(loc=1, ncol=4, shadow=True)
        ax.grid()
        fig.savefig('{}.reward_over_learning.svg'.format(self.config.prefix),
                    dpi=300, transparent=False, bbox_inches='tight')
        # plt.show()   
        matplotlib.pyplot.close('all')
    
    def reward_over_agents(self):
        average_average_reward = {}
        average_std_reward = {}
        for (dirpath, dirnames, _) in os.walk(self.config.dir):
            for dirname in dirnames:
                if self.config.pattern in dirname and 'ag' in dirname:
                    filename = os.sep.join([dirpath, dirname, 'logs/result.json'])
                    agents = dirname.split('_')[4].strip('ag')
                    LOGGER.info('Loading %s..', filename)
                    print(filename)
                    with open(filename, 'r') as jsonfile:
                        average = []
                        std = []
                        for checkpoint in jsonfile: # enumerate cannot be used due to the size of the file
                            complete = json.loads(checkpoint)
                            if complete['timesteps_total'] < 2000:
                                continue
                            if complete['timesteps_total'] > 3000:
                                break
                            _rewards = []
                            for policy in complete['policies'].values():
                                _rewards.append(policy['stats']['agent_reward'])
                            average.append(np.mean(_rewards))
                            std.append(np.std(_rewards))
                        average_average_reward[int(agents)] = np.mean(average)
                        average_std_reward[int(agents)] = np.mean(std)

        agents = sorted(average_average_reward.keys())
        mean = [average_average_reward[agent] for agent in agents]
        std = [average_std_reward[agent] for agent in agents]

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.plot(agents, mean, label='Average Mean Reward')
        ax.plot(agents, std, label='Average Std Reward')
        ax.set(xlabel='Number of Agents', ylabel='Reward',
            title='Reward over Number of Agents')
        ax.legend(loc=1, ncol=2, shadow=True)
        ax.grid()
        fig.savefig('{}.reward_over_agents.svg'.format(self.config.prefix),
                    dpi=300, transparent=False, bbox_inches='tight')
        plt.show()
        matplotlib.pyplot.close('all')


####################################################################################################

def _main():
    """ Process the RLLIB logs/result.json """

    config = _argument_parser()

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler = cProfile.Profile()
        profiler.enable()
    ## ========================              PROFILER              ======================== ##

    statistics = Statistics(config)
    statistics.reward_over_agents()
    LOGGER.info('Done')

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler.disable()
        results = io.StringIO()
        pstats.Stats(profiler, stream=results).sort_stats('cumulative').print_stats(50)
        LOGGER.info('Profiler: \n%s', pformat(results.getvalue()))
    ## ========================              PROFILER              ======================== ##


if __name__ == '__main__':
    _main()