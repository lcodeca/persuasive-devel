#!/usr/bin/env python3

""" Process a MULTIPLE RLLIB logs/result.json """

import json
import logging
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class StatAggrExps(object):
    """ Loads the result.json file as a time series. """

    def __init__(self, directory, pattern, prefix, begin, end):
        self.dir = directory
        self.pattern = pattern
        self.prefix = prefix
        self.begin = begin
        self.end = end
    
    def reward_over_agents(self):
        average_average_reward = {}
        average_std_reward = {}
        for (dirpath, dirnames, _) in os.walk(self.dir):
            for dirname in dirnames:
                if self.pattern in dirname and 'ag' in dirname:
                    filename = os.sep.join([dirpath, dirname, 'logs/result.json'])
                    agents = dirname.split('_')[4].strip('ag')
                    logging.info('Loading %s..', filename)
                    print(filename)
                    with open(filename, 'r') as jsonfile:
                        average = []
                        std = []
                        for checkpoint in jsonfile: # enumerate cannot be used due to the size of the file
                            complete = json.loads(checkpoint)
                            if complete['timesteps_total'] < self.begin:
                                continue
                            if complete['timesteps_total'] > self.end:
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
        fig.savefig('{}.reward_over_agents.svg'.format(self.prefix),
                    dpi=300, transparent=False, bbox_inches='tight')
        plt.show()
        matplotlib.pyplot.close('all')
