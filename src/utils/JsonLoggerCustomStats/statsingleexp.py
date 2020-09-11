#!/usr/bin/env python3

""" Process a SINGLE RLLIB logs/result.json """

import collections
import json
import logging
from pprint import pprint
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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

class StatSingleExp(object):
    """ Process a SINGLE RLLIB logs/result.json file as a time series. """

    def __init__(self, filename, prefix):
        self.input = filename
        self.prefix = prefix

    @staticmethod
    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    def reward_over_timesteps_total(self):
        logger.info('Computing the reward over the timesteps total.')
        x_coords = []
        y_coords = []
        median_y = []
        min_y = []
        max_y = []
        std_y = []
        with open(self.input, 'r') as jsonfile:
            for row in tqdm(jsonfile): # enumerate cannot be used due to the size of the file
                complete = json.loads(row)
                x_coords.append(complete['timesteps_total'])
                y_coords.append(np.nanmean(complete['hist_stats']['policy_unique_reward']))
                min_y.append(np.nanmin(complete['hist_stats']['policy_unique_reward']))
                max_y.append(np.nanmax(complete['hist_stats']['policy_unique_reward']))
                median_y.append(np.nanmedian(complete['hist_stats']['policy_unique_reward']))
                std_y.append(np.nanstd(complete['hist_stats']['policy_unique_reward']))

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.errorbar(x_coords, y_coords, yerr=std_y, capsize=5, label='Mean [std]', fmt='-o')
        ax.plot(x_coords, min_y, label='Min')
        ax.plot(x_coords, max_y, label='Max')
        ax.plot(x_coords, median_y, label='Median')
        ax.set(xlabel='Learning step', ylabel='Reward', title='Reward over time')
        ax.legend(loc=1, ncol=4, shadow=True)
        ax.grid()
        fig.savefig('{}.reward_over_learning.svg'.format(self.prefix),
                    dpi=300, transparent=False, bbox_inches='tight')
        #plt.show()
        # ZOOM IT
        ax.set_ylim(-20000, 0)
        # plt.show()
        fig.savefig('{}.bounded_reward_over_learning.svg'.format(self.prefix),
                    dpi=300, transparent=False, bbox_inches='tight')
        matplotlib.pyplot.close('all')

    def average_arrival_over_timesteps_total(self):
        logger.info('Computing the average arrival time over the timesteps total.')
        x_coords = []
        y_coords = []
        median_y = []
        min_y = []
        max_y = []
        std_y = []
        with open(self.input, 'r') as jsonfile:
            for row in tqdm(jsonfile): # enumerate cannot be used due to the size of the file
                complete = json.loads(row)
                x_coords.append(complete['timesteps_total'])
                arrivals = []
                for episode in complete['hist_stats']['info_by_agent']:
                    for info in episode.values():
                        arrivals.append(info['arrival']/3600)
                y_coords.append(np.nanmean(arrivals))
                min_y.append(np.nanmin(arrivals))
                max_y.append(np.nanmax(arrivals))
                median_y.append(np.nanmedian(arrivals))
                std_y.append(np.nanstd(arrivals))

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.errorbar(x_coords, y_coords, yerr=std_y, capsize=5, label='Mean [std]', fmt='-o')
        ax.plot(x_coords, min_y, label='Min')
        ax.plot(x_coords, max_y, label='Max')
        ax.plot(x_coords, median_y, label='Median')
        ax.set(xlabel='Learning step', ylabel='Time [h]', title='Arrival at destination over time.')
        ax.legend(loc=1, ncol=4, shadow=True)
        ax.grid()
        fig.savefig('{}.average_arrival_over_learning.svg'.format(self.prefix),
                    dpi=300, transparent=False, bbox_inches='tight')
        #plt.show()
        matplotlib.pyplot.close('all')

    def average_actions_over_episodes_total(self):
        logger.info('Computing the average number of actions over the episodes.')
        x_coords = []
        mean_y = []
        median_y = []
        min_y = []
        max_y = []
        std_y = []
        episodes = 0
        with open(self.input, 'r') as jsonfile:
            for row in tqdm(jsonfile): # enumerate cannot be used due to the size of the file
                complete = json.loads(row)
                episodes += complete['episodes_this_iter']
                x_coords.append(episodes)
                _actions = []
                for episode in complete['hist_stats']['rewards_by_agent']:
                    for agent_rewards in episode.values():
                        _actions.append(len(agent_rewards))
                min_y.append(np.nanmin(_actions))
                max_y.append(np.nanmax(_actions))
                median_y.append(np.nanmedian(_actions))
                mean_y.append(np.nanmean(_actions))
                std_y.append(np.nanstd(_actions))

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.errorbar(x_coords, mean_y, yerr=std_y, capsize=5, label='Mean [std]', fmt='-o')
        ax.plot(x_coords, min_y, label='Min')
        ax.plot(x_coords, max_y, label='Max')
        ax.plot(x_coords, median_y, label='Median')
        ax.set(xlabel='Episodes', ylabel='Actions', title='Actions per episode')
        ax.legend(loc=1, ncol=4, shadow=True)
        ax.grid()
        fig.savefig('{}.actions_over_episodes.svg'.format(self.prefix),
                    dpi=300, transparent=False, bbox_inches='tight')
        # plt.show()
        matplotlib.pyplot.close('all')

    def policy_loss_over_timesteps_total(self):
        logger.info('Computing the policy loss over the timesteps total.')
        x_coords = []
        y_coords = []
        with open(self.input, 'r') as jsonfile:
            for row in tqdm(jsonfile): # enumerate cannot be used due to the size of the file
                complete = json.loads(row)
                x_coords.append(complete['timesteps_total'])
                y_coords.append(complete['info']['learner']['unique']['policy_loss'])
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.plot(x_coords, y_coords)
        ax.set(xlabel='Learning step', ylabel='Loss', title='Policy Loss')
        ax.grid()
        fig.savefig('{}.policy_loss_over_learning.svg'.format(self.prefix),
                    dpi=300, transparent=False, bbox_inches='tight')
        # plt.show()
        matplotlib.pyplot.close('all')


    def policy_entropy_over_timesteps_total(self):
        logger.info('Computing the policy entropy over the timesteps total.')
        x_coords = []
        y_coords = []
        with open(self.input, 'r') as jsonfile:
            for row in tqdm(jsonfile): # enumerate cannot be used due to the size of the file
                complete = json.loads(row)
                x_coords.append(complete['timesteps_total'])
                y_coords.append(complete['info']['learner']['unique']['policy_entropy'])
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.plot(x_coords, y_coords)
        ax.set(xlabel='Learning step', ylabel='Entropy', title='Policy Entropy')
        ax.grid()
        fig.savefig('{}.policy_entropy_over_learning.svg'.format(self.prefix),
                    dpi=300, transparent=False, bbox_inches='tight')
        # plt.show()
        matplotlib.pyplot.close('all')

    def info_by_agent(self):
        logger.info('Computing the detailed info for each agent over the episodes.')
        agents = {}
        with open(self.input, 'r') as jsonfile:
            for row in tqdm(jsonfile): # enumerate cannot be used due to the size of the file
                complete = json.loads(row)
                # pprint(complete)
                info_by_episode = complete['hist_stats']['info_by_agent']
                last_action_by_agent = complete['hist_stats']['last_action_by_agent']
                rewards_by_agent = complete['hist_stats']['rewards_by_agent']
                for pos, episode in enumerate(info_by_episode):
                    for agent, info in episode.items():
                        # {'arrival': 26254.0,
                        #  'cost': 235.87030648896734,
                        #  'departure': 25886.0,
                        #  'discretized-cost': 2,
                        #  'ett': 235.8703064889596,
                        #  'ext': {'passenger': [237.20417391808218,
                        #                        235.87030648896734]},
                        #  'mode': 'passenger',
                        #  'rtt': 368.0,
                        #  'timeLoss': 98.62,
                        #  'wait': 6146.0}
                        if agent not in agents:
                            agents[agent] = {
                                'episode': [],
                                'reward': [],
                                'actions': [],
                                'mode': [],
                                ########################
                                'cost': [],
                                'discretized-cost': [],
                                'rtt': [],
                                'arrival': [],
                                'departure': [],
                                'ett': [],
                                'wait': [],
                                'timeLoss': [],
                                ########################
                                'difference': [],
                            }

                        agents[agent]['episode'].append(len(agents[agent]['episode']) + 1)
                        agents[agent]['reward'].append(sum(rewards_by_agent[pos][agent]))
                        agents[agent]['actions'].append(len(rewards_by_agent[pos][agent]))
                        agents[agent]['mode'].append(last_action_by_agent[pos][agent])
                        #############
                        agents[agent]['cost'].append(info['cost']/60.0)
                        agents[agent]['discretized-cost'].append(info['discretized-cost'])
                        agents[agent]['rtt'].append(info['rtt']/60.0)
                        #############
                        agents[agent]['arrival'].append(info['arrival']/3600.0)
                        agents[agent]['departure'].append(info['departure']/3600.0)
                        agents[agent]['ett'].append(info['ett']/60.0)
                        agents[agent]['wait'].append(info['wait']/60.0)
                        agents[agent]['timeLoss'].append(info['timeLoss'])
                        #############
                        agents[agent]['difference'].append((info['ett'] - info['rtt'])/60.0)

        for agent, stats in tqdm(agents.items()):
            fig, axs = plt.subplots(5, 2, sharex=True, figsize=(20, 20), constrained_layout=True)
            fig.suptitle('{}'.format(agent))

            rtt_max = np.nanmax(stats['rtt'])
            ett_max = np.nanmax(stats['ett'])
            ett_rtt_max = np.nanmax(np.array([ett_max, rtt_max]))
            if np.isnan(ett_rtt_max):
                ett_rtt_max = 1
            ett_rtt_max += ett_rtt_max * 0.1

            # Plot each graph
            axs[0][0].plot(stats['episode'], stats['reward'], label='Reward',
                           color='blue', marker='o', linestyle='solid', linewidth=2, markersize=8)
            axs[0][0].set_ylabel('Reward')
            axs[0][0].grid(True)

            axs[1][0].plot(stats['episode'], stats['actions'],
                           label='Number of actions', color='red', marker='o', linestyle='solid',
                           linewidth=2, markersize=8)
            axs[1][0].set_ylabel('Actions [#]')
            axs[1][0].grid(True)

            axs[2][0].plot(stats['episode'], stats['mode'],
                           label='Selected mode', color='green', marker='o', linestyle='solid',
                           linewidth=2, markersize=8)
            axs[4][0].set_ylim(0, np.nanmax(stats['mode']))
            axs[2][0].set_ylabel('Mode')
            axs[2][0].grid(True)

            axs[3][0].plot(stats['episode'], stats['ett'],
                           label='Estimated Travel Time', color='black', marker='o',
                           linestyle='solid', linewidth=2, markersize=8)
            axs[3][0].set_ylim(0, ett_rtt_max)
            axs[3][0].set_ylabel('Est TT [m]')
            axs[3][0].grid(True)

            axs[4][0].plot(stats['episode'], stats['rtt'],
                           label='Real Travel Time', color='magenta', marker='o', linestyle='solid',
                           linewidth=2, markersize=8)
            axs[4][0].set_ylim(0, ett_rtt_max)
            axs[4][0].set_ylabel('Real TT [m]')
            axs[4][0].set_xlabel('Episode [#]')
            axs[4][0].grid(True)

            axs[0][1].plot(stats['episode'], stats['departure'], 'b-',
                           label='Departure', color='blue', marker='o', linestyle='solid',
                           linewidth=2, markersize=8)
            axs[0][1].axhline(y=9.0, color='red', linestyle='dashed')
            axs[0][1].set_ylabel('Departure [h]')
            axs[0][1].grid(True)

            axs[1][1].plot(stats['episode'], stats['arrival'], 'r-',
                           label='Arrival', color='red', marker='o', linestyle='solid', linewidth=2,
                           markersize=8)
            axs[1][1].axhline(y=9.0, color='red', linestyle='dashed')
            axs[1][1].set_ylabel('Arrival [h]')
            axs[1][1].grid(True)

            axs[2][1].plot(stats['episode'], stats['wait'], 'g-',
                           label='Waiting at destination', color='green', marker='o',
                           linestyle='solid', linewidth=2, markersize=8)
            axs[2][1].axhline(y=0.0, color='red', linestyle='dashed')
            axs[2][1].set_ylabel('Wait @ destination [m]')
            axs[2][1].grid(True)

            axs[3][1].plot(stats['episode'], stats['cost'], 'k-',
                           label='Estimated cost', color='black', marker='o', linestyle='solid',
                           linewidth=2, markersize=8)
            axs[3][1].set_ylabel('Est Cost [m]')
            axs[3][1].grid(True)

            axs[4][1].plot(stats['episode'], stats['difference'], 'm-',
                           label='ETT / RTT Difference', color='magenta', marker='o',
                           linestyle='solid', linewidth=2, markersize=8)
            axs[4][1].axhline(y=0.0, color='red', linestyle='dashed')
            axs[4][1].set_ylabel('ETT / RTT Difference [m]')
            axs[4][1].set_xlabel('Episode [#]')
            axs[4][1].grid(True)

            fig.savefig('{}.{}.svg'.format(self.prefix, agent),
                        dpi=300, transparent=False, bbox_inches='tight')
            # fig.savefig('{}.{}.png'.format(self.prefix, agent),
            #             dpi=300, transparent=False, bbox_inches='tight')
            # plt.show()
            matplotlib.pyplot.close('all')
            # sys.exit()

    def additionals_by_agent(self):
        logger.info('Computing the detailed info for each agent over the episodes.')
        with open(self.input, 'r') as jsonfile:
            episodes = collections.defaultdict(list)
            for row in tqdm(jsonfile): # enumerate cannot be used due to the size of the file
                complete = json.loads(row)
                # pprint(complete)
                info_by_episode = complete['hist_stats']['info_by_agent']
                for episode in info_by_episode:
                    for agent, info_episode in episode.items():
                        episodes[agent].append(len(episodes[agent])+1)
                        if 'ext' not in info_episode:
                            continue
                        fig, ax = plt.subplots(3, 2, figsize=(15, 10))
                        fig.suptitle('ETT variation during episode {}'.format(len(episodes[agent])))
                        grid = {0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (0, 1), 4: (1, 1), 5: (2, 1)}
                        for pos, (mode, values) in enumerate(info_episode['ext'].items()):
                            x_coords = list(range(len(values)))
                            ax[grid[pos][0]][grid[pos][1]].plot(
                                x_coords, values, '-o', label='{}'.format(mode))
                            ax[grid[pos][0]][grid[pos][1]].legend()
                            ax[grid[pos][0]][grid[pos][1]].grid()
                        ax[2][0].set(xlabel='Learning steps')
                        ax[2][1].set(xlabel='Learning steps')
                        ax[0][0].set(ylabel='ETT[s]')
                        ax[1][0].set(ylabel='ETT[s]')
                        ax[2][0].set(ylabel='ETT[s]')
                        # plt.show()
                        fig.savefig(
                            '{}.{}.ett_over_episode_{}.svg'.format(
                                self.prefix, agent, len(episodes[agent])),
                            dpi=300, transparent=False, bbox_inches='tight')
                        matplotlib.pyplot.close('all')
