#!/usr/bin/env python3

""" Process the RLLIB logs/result.json """

import argparse
import collections
import cProfile
import io
import json
import logging
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
        '--input', required=True, type=str, help='Input JSON file.')
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

    def elapsed_episode_time_over_timesteps_total(self):
        LOGGER.info('Loading %s..', self.config.input)
        x_coords = []
        y_coords = []
        with open(self.config.input, 'r') as jsonfile:
            for row in jsonfile: # enumerate cannot be used due to the size of the file
                complete = json.loads(row)
                x_coords.append(complete['timesteps_total'])
                y_coords.append(complete['episode_elapsed_time_mean'])

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.plot(x_coords, y_coords)
        ax.set(xlabel='Learning step', ylabel='Time [s]',
            title='Average Episode Duration')
        ax.grid()
        fig.savefig('{}.episode_duration_over_learning.svg'.format(self.config.prefix),
                    dpi=300, transparent=False, bbox_inches='tight')
        # plt.show()   
        matplotlib.pyplot.close('all')

    def average_actions_over_episodes_total(self):
        LOGGER.info('Loading %s..', self.config.input)
        x_coords = []
        mean_y = []
        median_y = []
        min_y = []
        max_y = []
        std_y = []
        episodes = 0
        with open(self.config.input, 'r') as jsonfile:
            for row in jsonfile: # enumerate cannot be used due to the size of the file
                complete = json.loads(row)
                episodes += complete['episodes_this_iter']
                x_coords.append(episodes)
                _actions = []
                for policy in complete['policies'].values():
                    _actions.append(policy['stats']['actions_this_episode'])
                min_y.append(min(_actions))
                max_y.append(max(_actions))
                median_y.append(np.median(_actions))
                mean_y.append(np.mean(_actions))
                std_y.append(np.std(_actions))

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.errorbar(x_coords, mean_y, yerr=std_y, capsize=5, label='Mean [std]', fmt='-o')
        ax.plot(x_coords, min_y, label='Min')
        ax.plot(x_coords, max_y, label='Max')
        ax.plot(x_coords, median_y, label='Median')
        ax.set(xlabel='Episodes', ylabel='Actions',
            title='Actions per episode')
        ax.legend(loc=1, ncol=4, shadow=True)
        ax.grid()
        fig.savefig('{}.actions_over_episodes.svg'.format(self.config.prefix),
                    dpi=300, transparent=False, bbox_inches='tight')
        # plt.show()   
        matplotlib.pyplot.close('all')

    def sequence_by_agent(self):
        LOGGER.info('Loading %s..', self.config.input)
        agents = {}
        with open(self.config.input, 'r') as jsonfile:
            for row in jsonfile: # enumerate cannot be used due to the size of the file
                complete = json.loads(row)
                for agent, policy in complete['policies'].items():
                    if agent not in agents:
                        agents[agent] = {
                            'episode': [],
                            'reward': [],
                            'actions': [],
                            'mode': [],
                        }
                    for sequence in policy['stats']['sequence']:
                        actions = []
                        rewards = []
                        for step in sequence:
                            _, action, _, reward = step
                            actions.append(action)
                            rewards.append(reward)
                        agents[agent]['episode'].append(len(agents[agent]['episode']) + 1)
                        agents[agent]['reward'].append(sum(rewards))
                        agents[agent]['actions'].append(len(actions))
                        agents[agent]['mode'].append(actions[-1])

        for agent, stats in agents.items():
            # https://matplotlib.org/3.1.1/gallery/ticks_and_spines/multiple_yaxis_with_spines.html
            fig, host = plt.subplots(figsize=(15, 10))
            fig.subplots_adjust(right=0.75)
            par1 = host.twinx()
            par2 = host.twinx()
            # Offset the right spine of par2.  The ticks and label have already been
            # placed on the right by twinx above.
            par2.spines['right'].set_position(('axes', 1.2))
            # Having been created by twinx, par2 has its frame off, so the line of its
            # detached spine is invisible.  First, activate the frame but make the patch
            # and spines invisible.
            make_patch_spines_invisible(par2)
            # Second, show the right spine.
            par2.spines['right'].set_visible(True)

            p1, = host.plot(stats['episode'], stats['reward'], 'b-', label='Reward')
            p2, = par1.plot(stats['episode'], stats['actions'], 'r-', label='Number of actions')
            p3, = par2.plot(stats['episode'], stats['mode'], 'g-', label='Selected mode')

            # host.set_xlim(0, 2)
            # host.set_ylim(0, 2)
            # par1.set_ylim(0, 4)
            # par2.set_ylim(1, 65)

            host.set_title('{}'.format(agent))
            host.set_xlabel('Episode')
            host.set_ylabel('Reward')
            par1.set_ylabel('Number of actions')
            par2.set_ylabel('Selected mode')

            host.yaxis.label.set_color(p1.get_color())
            par1.yaxis.label.set_color(p2.get_color())
            par2.yaxis.label.set_color(p3.get_color())

            tkw = dict(size=4, width=1.5)
            host.tick_params(axis='y', colors=p1.get_color(), **tkw)
            par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
            par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
            host.tick_params(axis='x', **tkw)

            lines = [p1, p2, p3]
            host.legend(lines, [l.get_label() for l in lines], loc=0, shadow=True)

            fig.savefig('{}.{}.svg'.format(self.config.prefix, agent), 
                        dpi=300, transparent=False, bbox_inches='tight')
            # plt.show()
            matplotlib.pyplot.close('all')     
            # sys.exit()
    
    def estimations_by_agent(self):
        LOGGER.info('Loading %s..', self.config.input)
        agents = {}
        with open(self.config.input, 'r') as jsonfile:
            for row in jsonfile: # enumerate cannot be used due to the size of the file
                complete = json.loads(row)
                for agent, policy in complete['policies'].items():
                    if agent not in agents:
                        agents[agent] = {
                            'episode': [],
                            'reward': [],
                            'actions': [],
                            'mode': [],
                            'cost': [],
                            'discretized-cost': [],
                            'rtt': [],
                            'state': []
                        }
                    for sequence in policy['stats']['sequence']:
                        actions = []
                        rewards = []
                        for step in sequence:
                            _, action, _, reward = step
                            actions.append(action)
                            rewards.append(reward)
                        agents[agent]['episode'].append(len(agents[agent]['episode']) + 1)
                        agents[agent]['reward'].append(sum(rewards))
                        agents[agent]['actions'].append(len(actions))
                        agents[agent]['mode'].append(actions[-1])
                    for info in policy['stats']['info']:
                        info = info[0]
                        agents[agent]['cost'].append(info['cost'])
                        agents[agent]['discretized-cost'].append(info['discretized-cost'])
                        agents[agent]['rtt'].append(info['rtt'])
                        agents[agent]['state'].append(info['from-state']['ett'])

        for agent, stats in agents.items():
            # https://matplotlib.org/gallery/subplots_axes_and_figures/ganged_plots.html#sphx-glr-gallery-subplots-axes-and-figures-ganged-plots-py
            fig, axs = plt.subplots(5, 1, sharex=True, figsize=(10, 20))
            # Reduce the horizontal space between axes
            fig.subplots_adjust(hspace=0.1)

            # Plot each graph
            axs[0].set_title('{}'.format(agent))
            axs[0].plot(stats['episode'], stats['reward'], 'b-', label='Reward')
            axs[0].set_ylabel('Reward')
            # axs[0].set_yticks(np.arange(-0.9, 1.0, 0.4))
            # axs[0].set_ylim(-1, 1)
            axs[1].plot(stats['episode'], stats['actions'], 'r-', label='Number of actions')
            axs[1].set_ylabel('Actions [#]')
            axs[2].plot(stats['episode'], stats['mode'], 'g-', label='Selected mode')
            axs[2].set_ylabel('Mode')
            axs[3].plot(stats['episode'], stats['cost'], 'k-', label='Estimated cost')
            axs[3].set_ylabel('Est Cost [s]')
            axs[4].plot(stats['episode'], stats['rtt'], 'm-', label='Real Travel Time')
            axs[4].set_ylabel('Real TT [s]')
            axs[4].set_xlabel('Episode [#]')

            fig.savefig('{}.{}.svg'.format(self.config.prefix, agent), 
                        dpi=300, transparent=False, bbox_inches='tight')
            # plt.show()
            matplotlib.pyplot.close('all')     
            # sys.exit()

    def additionals_by_agent(self):
        LOGGER.info('Loading %s..', self.config.input)
        with open(self.config.input, 'r') as jsonfile:
            episodes = collections.defaultdict(list)
            for row in jsonfile: # enumerate cannot be used due to the size of the file
                complete = json.loads(row)
                for agent, policy in complete['policies'].items():
                    for info_checkpoint in policy['stats']['info']:
                        for info_episode in info_checkpoint:
                            episodes[agent].append(len(episodes[agent])+1)
                            if 'ext' not in  info_episode:
                                continue
                            for mode, values in info_episode['ext'].items():
                                x_coords = list(range(len(values)))
                                fig, ax = plt.subplots(figsize=(15, 10))
                                ax.plot(x_coords, values, label='ETT')
                                ax.set(xlabel='Learning steps', ylabel='Time [s]', 
                                       title='ETT variation during an episode.')
                                ax.grid()
                                fig.savefig(
                                    '{}.{}.{}.ett_over_episode_{}.svg'.format(
                                        self.config.prefix, agent, mode, len(episodes[agent])),
                                    dpi=300, transparent=False, bbox_inches='tight')
                                # plt.show()   
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
    statistics.reward_over_timesteps_total()
    statistics.elapsed_episode_time_over_timesteps_total()
    statistics.average_actions_over_episodes_total()
    # statistics.sequence_by_agent()
    statistics.estimations_by_agent()
    statistics.additionals_by_agent()
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