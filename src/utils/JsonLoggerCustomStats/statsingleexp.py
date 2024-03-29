#!/usr/bin/env python3

""" Process a SINGLE RLLIB logs/result.json """

import collections
import json
import logging
from pprint import pprint
import sys
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

    def __init__(self, filename, prefix, max_agents=None, last=None, evaluation=False):
        self.input = filename
        self.prefix = prefix
        self.max = max_agents
        self.last = last
        self.evaluation = evaluation
        self.agents = None

    @staticmethod
    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    def recursive_print(self, dictionary):
        print(dictionary.keys())
        input('Press any key...')
        for key, value in dictionary.items():
            if isinstance(value, dict):
                print(key)
                self.recursive_print(value)
            else:
                print(value, key)
                input('Press any key...')

    def perf_over_timesteps_total(self):
        logger.info('Computing the system performances over the timesteps total.')
        perfs = collections.defaultdict(list)
        with open(self.input, 'r') as jsonfile:
            for row in tqdm(jsonfile): # enumerate cannot be used due to the size of the file
                complete = json.loads(row)
                for metric, value in complete['perf'].items():
                    perfs[metric].append((complete['timesteps_total'], value))

        fig, ax = plt.subplots(figsize=(15, 10))
        for metric, values in perfs.items():
            xval = []
            yval = []
            for timestep, val in values:
                xval.append(timestep)
                yval.append(val)
            ax.plot(xval, yval, label=metric)
        ax.set(xlabel='Timesteps', ylabel='Load [%]', title='System Performances')
        ax.legend(loc='best', shadow=True)
        ax.grid()
        fig.savefig('{}.sysperf_over_learning.svg'.format(self.prefix),
                    dpi=300, transparent=False, bbox_inches='tight')
        #plt.show()
        matplotlib.pyplot.close('all')


    def reward_over_timesteps_total(self):
        logger.info('Computing the reward over the timesteps total.')
        x_coords = []
        y_coords = []
        median_y = []
        min_y = []
        max_y = []
        std_y = []
        with open(self.input, 'r') as jsonfile:
            counter = 0
            for row in tqdm(jsonfile): # enumerate cannot be used due to the size of the file
                complete = json.loads(row)
                if self.evaluation:
                    if 'evaluation' in complete:
                        tmp = complete['evaluation']
                        tmp['timesteps_total'] = complete['timesteps_total']
                        complete = tmp
                    else:
                        # evaluation stats requested but not present in the results
                        continue
                if 'policy_unique_reward' in complete['hist_stats']:
                    x_coords.append(complete['timesteps_total'])
                    y_coords.append(np.nanmean(complete['hist_stats']['policy_unique_reward']))
                    min_y.append(np.nanmin(complete['hist_stats']['policy_unique_reward']))
                    max_y.append(np.nanmax(complete['hist_stats']['policy_unique_reward']))
                    median_y.append(np.nanmedian(complete['hist_stats']['policy_unique_reward']))
                    std_y.append(np.nanstd(complete['hist_stats']['policy_unique_reward']))
                else:
                    logger.critical('Missing stats in row %d', counter)
                counter += 1

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.errorbar(x_coords, y_coords, yerr=std_y, capsize=5, label='Mean [std]', fmt='-o')
        ax.plot(x_coords, min_y, label='Min')
        ax.plot(x_coords, max_y, label='Max')
        ax.plot(x_coords, median_y, label='Median')
        ax.set(xlabel='Learning step', ylabel='Reward', title='Reward over time')
        ax.legend(loc='best', ncol=4, shadow=True)
        ax.grid()
        fig.savefig('{}.reward_over_learning.svg'.format(self.prefix),
                    dpi=300, transparent=False, bbox_inches='tight')
        #plt.show()
        # ZOOM IT
        # ax.set_ylim(-20000, 0)
        # # plt.show()
        # fig.savefig('{}.bounded_reward_over_learning.svg'.format(self.prefix),
        #             dpi=300, transparent=False, bbox_inches='tight')
        matplotlib.pyplot.close('all')

    def total_reward_over_timesteps_total(self):
        logger.info('Computing the reward over the timesteps total.')
        x_coords = []
        y_coords = []
        min_y = []
        max_y = []
        with open(self.input, 'r') as jsonfile:
            counter = 0
            for row in tqdm(jsonfile): # enumerate cannot be used due to the size of the file
                complete = json.loads(row)
                if self.evaluation:
                    if 'evaluation' in complete:
                        tmp = complete['evaluation']
                        tmp['timesteps_total'] = complete['timesteps_total']
                        complete = tmp
                    else:
                        # evaluation stats requested but not present in the results
                        continue

                x_coords.append(complete['timesteps_total'])
                y_coords.append(complete['episode_reward_mean'])
                min_y.append(complete['episode_reward_min'])
                max_y.append(complete['episode_reward_max'])
                counter += 1

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.errorbar(x_coords, y_coords, capsize=5, label='Mean', fmt='-o')
        ax.plot(x_coords, min_y, label='Min')
        ax.plot(x_coords, max_y, label='Max')
        ax.set(xlabel='Learning step', ylabel='Reward', title='Reward over time')
        ax.legend(loc='best', ncol=4, shadow=True)
        ax.grid()
        fig.savefig('{}.total_reward_over_learning.svg'.format(self.prefix),
                    dpi=300, transparent=False, bbox_inches='tight')
        #plt.show()
        # ZOOM IT
        # ax.set_ylim(-20000, 0)
        # # plt.show()
        # fig.savefig('{}.bounded_reward_over_learning.svg'.format(self.prefix),
        #             dpi=300, transparent=False, bbox_inches='tight')
        matplotlib.pyplot.close('all')

    def missing_agents_over_timesteps_total(self):
        logger.info('Computing the missing agents over the timesteps total.')
        x_coords = []
        y_coords = []
        min_y = []
        max_y = []
        with open(self.input, 'r') as jsonfile:
            counter = 0
            for row in tqdm(jsonfile): # enumerate cannot be used due to the size of the file
                complete = json.loads(row)
                if self.evaluation:
                    if 'evaluation' in complete:
                        tmp = complete['evaluation']
                        tmp['timesteps_total'] = complete['timesteps_total']
                        complete = tmp
                    else:
                        # evaluation stats requested but not present in the results
                        continue

                x_coords.append(complete['timesteps_total'])
                y_coords.append(complete['custom_metrics']['episode_missing_agents_mean'])
                min_y.append(complete['custom_metrics']['episode_missing_agents_min'])
                max_y.append(complete['custom_metrics']['episode_missing_agents_max'])
                counter += 1

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.errorbar(x_coords, y_coords, capsize=5, label='Mean', fmt='-o')
        ax.plot(x_coords, min_y, label='Min')
        ax.plot(x_coords, max_y, label='Max')
        ax.set(xlabel='Learning step', ylabel='Missing Agents', title='Missing Agents Over Time')
        ax.legend(loc='best', ncol=4, shadow=True)
        ax.grid()
        fig.savefig('{}.missing_agents_over_learning.svg'.format(self.prefix),
                    dpi=300, transparent=False, bbox_inches='tight')
        #plt.show()
        matplotlib.pyplot.close('all')

    def on_time_agents_over_timesteps_total(self):
        logger.info('Computing the on-time agents over the timesteps total.')
        x_coords = []
        y_coords = []
        min_y = []
        max_y = []
        with open(self.input, 'r') as jsonfile:
            counter = 0
            for row in tqdm(jsonfile): # enumerate cannot be used due to the size of the file
                complete = json.loads(row)
                if self.evaluation:
                    if 'evaluation' in complete:
                        tmp = complete['evaluation']
                        tmp['timesteps_total'] = complete['timesteps_total']
                        complete = tmp
                    else:
                        # evaluation stats requested but not present in the results
                        continue

                x_coords.append(complete['timesteps_total'])
                y_coords.append(complete['custom_metrics']['episode_on_time_agents_mean'])
                min_y.append(complete['custom_metrics']['episode_on_time_agents_min'])
                max_y.append(complete['custom_metrics']['episode_on_time_agents_max'])
                counter += 1

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.errorbar(x_coords, y_coords, capsize=5, label='Mean', fmt='-o')
        ax.plot(x_coords, min_y, label='Min')
        ax.plot(x_coords, max_y, label='Max')
        ax.set(xlabel='Learning step', ylabel='On-time Agents', title='On-time Agents Over Time')
        ax.legend(loc='best', ncol=4, shadow=True)
        ax.grid()
        fig.savefig('{}.on_time_agents_over_learning.svg'.format(self.prefix),
                    dpi=300, transparent=False, bbox_inches='tight')
        #plt.show()
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
            counter = 0
            for row in tqdm(jsonfile): # enumerate cannot be used due to the size of the file
                complete = json.loads(row)
                if self.evaluation:
                    if 'evaluation' in complete:
                        tmp = complete['evaluation']
                        tmp['timesteps_total'] = complete['timesteps_total']
                        complete = tmp
                    else:
                        # evaluation stats requested but not present in the results
                        continue
                if 'info_by_agent' in complete['hist_stats']:
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
                else:
                    logger.critical('Missing stats in row %d', counter)
                counter += 1

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.errorbar(x_coords, y_coords, yerr=std_y, capsize=5, label='Mean [std]', fmt='-o')
        ax.plot(x_coords, min_y, label='Min')
        ax.plot(x_coords, max_y, label='Max')
        ax.plot(x_coords, median_y, label='Median')
        ax.set(xlabel='Learning step', ylabel='Time [h]', title='Arrival at destination over time.')
        ax.legend(loc='best', ncol=4, shadow=True)
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
            counter = 0
            for row in tqdm(jsonfile): # enumerate cannot be used due to the size of the file
                complete = json.loads(row)
                if self.evaluation:
                    if 'evaluation' in complete:
                        complete = complete['evaluation']
                    else:
                        # evaluation stats requested but not present in the results
                        continue
                if 'rewards_by_agent' in complete['hist_stats']:
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
                else:
                    logger.critical('Missing stats in row %d', counter)
                counter += 1

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.errorbar(x_coords, mean_y, yerr=std_y, capsize=5, label='Mean [std]', fmt='-o')
        ax.plot(x_coords, min_y, label='Min')
        ax.plot(x_coords, max_y, label='Max')
        ax.plot(x_coords, median_y, label='Median')
        ax.set(xlabel='Episodes', ylabel='Actions', title='Actions per episode')
        ax.legend(loc='best', ncol=4, shadow=True)
        ax.grid()
        fig.savefig('{}.actions_over_episodes.svg'.format(self.prefix),
                    dpi=300, transparent=False, bbox_inches='tight')
        # plt.show()
        matplotlib.pyplot.close('all')

    def qvalues_over_timesteps_total(self):
        logger.info('Computing the qvalues over the timesteps total.')
        x_coords = []
        y_coords = []
        min_y = []
        max_y = []
        with open(self.input, 'r') as jsonfile:
            for row in tqdm(jsonfile): # enumerate cannot be used due to the size of the file
                complete = json.loads(row)
                x_coords.append(complete['timesteps_total'])
                y_coords.append(complete['info']['learner']['unique']['mean_q'])
                min_y.append(complete['info']['learner']['unique']['min_q'])
                max_y.append(complete['info']['learner']['unique']['max_q'])
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.plot(x_coords, y_coords, label='Mean')
        ax.plot(x_coords, min_y, label='Min')
        ax.plot(x_coords, max_y, label='Max')
        ax.set(xlabel='Learning step', ylabel='Q-Values', title='Q-Values')
        ax.grid()
        fig.savefig('{}.qvalues_over_learning.svg'.format(self.prefix),
                    dpi=300, transparent=False, bbox_inches='tight')
        # plt.show()
        matplotlib.pyplot.close('all')

    def td_error_over_timesteps_total(self):
        logger.info('Computing the TD error over the timesteps total.')
        x_coords = []
        y_coords = []
        with open(self.input, 'r') as jsonfile:
            for row in tqdm(jsonfile): # enumerate cannot be used due to the size of the file
                complete = json.loads(row)
                x_coords.append(complete['timesteps_total'])
                y_coords.append(complete['info']['learner']['unique']['mean_td_error'])
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.plot(x_coords, y_coords)
        ax.set(xlabel='Learning step', ylabel='Loss', title='Mean TD Error')
        ax.grid()
        fig.savefig('{}.td_error_over_learning.svg'.format(self.prefix),
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
                if self.evaluation:
                    if 'evaluation' in complete:
                        complete = complete['evaluation']
                    else:
                        # evaluation stats requested but not present in the results
                        continue
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

    def total_loss_over_timesteps_total(self):
        logger.info('Computing the total loss over the timesteps total.')
        x_coords = []
        y_coords = []
        with open(self.input, 'r') as jsonfile:
            for row in tqdm(jsonfile): # enumerate cannot be used due to the size of the file
                complete = json.loads(row)
                if self.evaluation:
                    if 'evaluation' in complete:
                        complete = complete['evaluation']
                    else:
                        # evaluation stats requested but not present in the results
                        continue
                x_coords.append(complete['timesteps_total'])
                y_coords.append(complete['info']['learner']['unique']['total_loss'])
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.plot(x_coords, y_coords)
        ax.set(xlabel='Learning step', ylabel='Loss', title='Total Loss')
        ax.grid()
        fig.savefig('{}.total_loss_over_learning.svg'.format(self.prefix),
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
                if self.evaluation:
                    if 'evaluation' in complete:
                        complete = complete['evaluation']
                    else:
                        # evaluation stats requested but not present in the results
                        continue
                x_coords.append(complete['timesteps_total'])
                y_coords.append(complete['info']['learner']['unique']['entropy'])
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.plot(x_coords, y_coords)
        ax.set(xlabel='Learning step', ylabel='Entropy', title='Policy Entropy')
        ax.grid()
        fig.savefig('{}.policy_entropy_over_learning.svg'.format(self.prefix),
                    dpi=300, transparent=False, bbox_inches='tight')
        # plt.show()
        matplotlib.pyplot.close('all')

    def _add_agent(self, agent):
        self.agents[agent] = {
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
            ########################
            'difference': [],
        }

    def _try_add_agent(self, agent):
        logger.debug('Trying to add agent %s, if possible', agent)
        if agent in self.agents:
            return False
        if self.max is None:
            self._add_agent(agent)
            return True
        if len(self.agents) < self.max:
            self._add_agent(agent)
            return True
        return False

    def info_by_agent(self):
        logger.info('Computing the detailed info for each agent over the episodes.')
        self.agents = {}
        with open(self.input, 'r') as jsonfile:
            counter = 0
            for row in tqdm(jsonfile): # enumerate cannot be used due to the size of the file
                complete = json.loads(row)
                if self.evaluation:
                    if 'evaluation' in complete:
                        complete = complete['evaluation']
                    else:
                        # evaluation stats requested but not present in the results
                        continue
                try:
                    info_by_episode = complete['hist_stats']['info_by_agent']
                    last_action_by_agent = complete['hist_stats']['last_action_by_agent']
                    rewards_by_agent = complete['hist_stats']['rewards_by_agent']
                    for pos, episode in enumerate(info_by_episode):
                        for agent, info in episode.items():
                            self._try_add_agent(agent)
                            if agent not in self.agents:
                                continue
                            self.agents[agent]['episode'].append(
                                len(self.agents[agent]['episode']) + 1)
                            self.agents[agent]['reward'].append(sum(rewards_by_agent[pos][agent]))
                            self.agents[agent]['actions'].append(len(rewards_by_agent[pos][agent]))
                            self.agents[agent]['mode'].append(last_action_by_agent[pos][agent])
                            #############
                            self.agents[agent]['cost'].append(info['cost']/60.0)
                            self.agents[agent]['discretized-cost'].append(info['discretized-cost'])
                            self.agents[agent]['rtt'].append(info['rtt']/60.0)
                            #############
                            self.agents[agent]['arrival'].append(info['arrival']/3600.0)
                            self.agents[agent]['departure'].append(info['departure']/3600.0)
                            self.agents[agent]['ett'].append(info['ett']/60.0)
                            self.agents[agent]['wait'].append(info['wait']/60.0)
                            #############
                            self.agents[agent]['difference'].append(
                                (info['ett'] - info['rtt'])/60.0)
                except KeyError:
                    logger.critical('Missing stats in row %d', counter)
                counter += 1

        for agent, stats in tqdm(self.agents.items()):
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
        if self.last:
            self.last_additionals_by_agent()
        else:
            self.all_additionals_by_agent()

    def _try_insert_agent(self, agent):
        logger.debug('Trying to add agent %s, if possible', agent)
        if agent in self.agents:
            return False
        if self.max is None:
            self.agents.append(agent)
            return True
        if len(self.agents) < self.max:
            self.agents.append(agent)
            return True
        return False

    def all_additionals_by_agent(self):
        logger.info('Computing the detailed info for each agent over the episodes.')
        with open(self.input, 'r') as jsonfile:
            self.agents = []
            episodes = collections.defaultdict(int)
            counter = 0
            for row in tqdm(jsonfile): # enumerate cannot be used due to the size of the file
                complete = json.loads(row)
                if self.evaluation:
                    if 'evaluation' in complete:
                        complete = complete['evaluation']
                    else:
                        # evaluation stats requested but not present in the results
                        continue
                try:
                    info_by_episode = complete['hist_stats']['info_by_agent']
                    for episode in info_by_episode:
                        for agent, info_episode in episode.items():
                            if 'ext' not in info_episode:
                                continue
                            self._try_insert_agent(agent)
                            if agent not in self.agents:
                                continue
                            episodes[agent] += 1
                            fig, ax = plt.subplots(3, 2, figsize=(15, 10))
                            fig.suptitle('ETT variation during episode {}'.format(episodes[agent]))
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
                                    self.prefix, agent, episodes[agent]),
                                dpi=300, transparent=False, bbox_inches='tight')
                            matplotlib.pyplot.close('all')
                except KeyError:
                    logger.critical('Missing stats in row %d', counter)
                counter += 1

    def last_additionals_by_agent(self):
        logger.info('Computing the detailed info for each agent over the episodes.')
        with open(self.input, 'r') as jsonfile:
            self.agents = []
            episodes = collections.defaultdict(int)
            infos = collections.defaultdict(dict)
            counter = 0
            for row in tqdm(jsonfile): # enumerate cannot be used due to the size of the file
                complete = json.loads(row)
                if self.evaluation:
                    if 'evaluation' in complete:
                        complete = complete['evaluation']
                    else:
                        # evaluation stats requested but not present in the results
                        continue
                try:
                    info_by_episode = complete['hist_stats']['info_by_agent']
                    for episode in info_by_episode:
                        for agent, info_episode in episode.items():
                            if 'ext' not in info_episode:
                                continue
                            self._try_insert_agent(agent)
                            if agent not in self.agents:
                                continue
                            infos[agent] = info_episode['ext']
                            episodes[agent] += 1
                except KeyError:
                    logger.critical('Missing stats in row %d', counter)
                counter += 1

        for agent in tqdm(self.agents):
            fig, ax = plt.subplots(3, 2, figsize=(15, 10))
            fig.suptitle('ETT variation during episode {}'.format(episodes[agent]))
            grid = {0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (0, 1), 4: (1, 1), 5: (2, 1)}
            for pos, (mode, values) in enumerate(infos[agent].items()):
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
                    self.prefix, agent, episodes[agent]),
                dpi=300, transparent=False, bbox_inches='tight')
            matplotlib.pyplot.close('all')

    # def epsilon_over_timesteps_total(self):
    #     logger.info('Computing the exploration epsilon over the timesteps total.')
    #     x_coords = []
    #     y_coords = []
    #     with open(self.input, 'r') as jsonfile:
    #         for row in tqdm(jsonfile): # enumerate cannot be used due to the size of the file
    #             complete = json.loads(row)
    #             pprint(complete['info'])
    #             exit(666)
    #             x_coords.append(complete['timesteps_total'])
    #             y_coords.append(complete['info']['learner']['unique']['policy_entropy'])
    #     fig, ax = plt.subplots(figsize=(15, 10))
    #     ax.plot(x_coords, y_coords)
    #     ax.set(xlabel='Learning step', ylabel='Entropy', title='Policy Entropy')
    #     ax.grid()
    #     fig.savefig('{}.policy_entropy_over_learning.svg'.format(self.prefix),
    #                 dpi=300, transparent=False, bbox_inches='tight')
    #     # plt.show()
    #     matplotlib.pyplot.close('all')
