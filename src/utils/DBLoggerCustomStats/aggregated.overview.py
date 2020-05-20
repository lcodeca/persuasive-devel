#!/usr/bin/env python3

""" Process the DBLogger directory structure generating an aggregated overview. """

import argparse
import collections
import cProfile
import io
import json
import logging
import os
from pprint import pformat, pprint
import pstats
import re
import sys

from deepdiff import DeepDiff
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from dbloggerstats import DBLoggerStats

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

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

def _argument_parser():
    """ Argument parser for the stats parser. """
    parser = argparse.ArgumentParser(
        description='RLLIB & SUMO Statistics parser.')
    parser.add_argument(
        '--dir-tree', required=True, type=str, 
        help='DBLogger directory.')
    parser.add_argument(
        '--graph', required=True, 
        help='Output prefix for the graph.')
    parser.add_argument(
        '--data', required=True, 
        help='Input/Output file for the processed data.')
    parser.add_argument(
        '--window', default=10, type=int, 
        help='Number of training runs to consider for policy stability.')
    parser.add_argument(
        '--profiler', dest='profiler', action='store_true', help='Enable cProfile.')
    parser.set_defaults(profiler=False)
    return parser.parse_args()

def _main():
    """ Process the DBLogger directory structure generating an aggregated overview. """

    config = _argument_parser()

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler = cProfile.Profile()
        profiler.enable()
    ## ========================              PROFILER              ======================== ##

    statistics = AggregatedOverview(config.dir_tree, config.data, config.window, config.graph)
    statistics.aggregate_data()
    statistics.generate_plot()
    LOGGER.info('Done')

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler.disable()
        results = io.StringIO()
        pstats.Stats(profiler, stream=results).sort_stats('cumulative').print_stats(50)
        LOGGER.info('Profiler: \n%s', pformat(results.getvalue()))
    ## ========================              PROFILER              ======================== ##

class AggregatedOverview(DBLoggerStats):

    def __init__(self, directory, dataset, window, prefix):
        super().__init__(directory)
        self.dataset_fname = dataset
        self.stability_window = window
        self.output_prefix = prefix
        self.aggregated_dataset = dict()

    ####################################### DATA  AGGREGATOR #######################################

    def _init_datastructure(self):
        """ Loads the dataset from file if exist or generates the empty structure. """

        if os.path.exists(self.dataset_fname):
            with open(self.dataset_fname, 'r') as jsonfile:
                self.aggregated_dataset = json.load(jsonfile)
        else: 
            # main graph: reward over learning
            self.aggregated_dataset['reward-min'] = list()
            self.aggregated_dataset['reward-max'] = list()
            self.aggregated_dataset['reward-mean'] = list()
            self.aggregated_dataset['reward-median'] = list()
            self.aggregated_dataset['reward-std'] = list()
            self.aggregated_dataset['learning-step'] = list()

            # waiting and policy stability by agent
            self.aggregated_dataset['waited'] = dict()
            self.aggregated_dataset['stable'] = dict()
            self.aggregated_dataset['percentage-waited'] = list()
            self.aggregated_dataset['percentage-stable'] = list()

            # aggregation
            self.aggregated_dataset['training-folders'] = list()
        
        LOGGER.debug('Aggregated data structure: \n%s', pformat(self.aggregated_dataset))
    
    def _save_satastructure(self):
        """ Saves the datastructure to file. """
        with open(self.dataset_fname, 'w') as jsonfile:
            json.dump(self.aggregated_dataset, jsonfile, indent=2)

    def aggregate_data(self):
        self._init_datastructure()
        # process the directory tree
        available_training_runs = self.alphanumeric_sort(os.listdir(self.dir))
        print('Processing the training runs...')
        for training_run in tqdm(available_training_runs):
            if training_run in self.aggregated_dataset['training-folders']:
                continue
            print('Processing {}/{}'.format(self.dir, training_run))
            agents, episodes, _ = self.get_training_components(training_run)
            
            # compute the reward
            rewards = list()
            for agent in agents:
                rewards.append(self.get_reward(training_run, agent))
            self.aggregated_dataset['reward-min'].append(min(rewards))
            self.aggregated_dataset['reward-max'].append(max(rewards))
            self.aggregated_dataset['reward-mean'].append(np.mean(rewards))
            self.aggregated_dataset['reward-median'].append(np.median(rewards))
            self.aggregated_dataset['reward-std'].append(np.std(rewards))

            # retrieve the learning steps
            step = self.get_timesteps_this_iter(training_run)
            if self.aggregated_dataset['learning-step']:
                step += self.aggregated_dataset['learning-step'][-1]
            self.aggregated_dataset['learning-step'].append(step)

            # process the episodes for "waiting too long" event
            for episode in episodes:
                for agent in agents:
                    if self._agent_waited_too_long(agent):
                        continue
                    last_action = self.get_last_action(training_run, episode, agent)
                    if last_action is 0:
                        self.aggregated_dataset['waited'][agent] = step
                    else:
                        self.aggregated_dataset['waited'][agent] = None
            waited = 0
            for agent in agents:
                if self.aggregated_dataset['waited'][agent] is not None:
                    waited += 1
            # waited : all_agents = x : 100 
            self.aggregated_dataset['percentage-waited'].append(waited * 100.0 / len(agents)) 

            # process the agents to define policy stability
            current_run_index = available_training_runs.index(training_run)
            window = []
            if current_run_index + 1 > self.stability_window:
                window = available_training_runs[current_run_index-self.stability_window:current_run_index]
            # print(current_run_index, self.stability_window, window)
            stables = 0
            for agent in agents:
                previous_policy = None
                stable = False
                # print(window)
                for run in window:
                    current_policy = self.get_best_actions(run, agent)
                    if previous_policy is None:
                        previous_policy = current_policy
                        continue
                    if DeepDiff(previous_policy, current_policy):
                        stable = False
                        break
                    else:
                        stable = True
                        # print(agent, run)
                self.aggregated_dataset['stable'][agent] = stable
                if stable:
                    stables +=1
            # stables : all_agents = x : 100 
            self.aggregated_dataset['percentage-stable'].append(stables * 100.0 / len(agents)) 

            self.aggregated_dataset['training-folders'].append(training_run)

        LOGGER.debug('UPDATED aggregated data structure: \n%s', pformat(self.aggregated_dataset))

        # save the new dataset into the dataset file
        self._save_satastructure() 

    ######################################## PLOT GENERATOR ########################################

    def generate_plot(self):
        print('Plot generation...')
        fig, axs = plt.subplots(
            3, sharex=True, figsize=(15, 15), constrained_layout=True, 
            gridspec_kw={'height_ratios': [4, 1, 1]})

        axs[0].errorbar(
            self.aggregated_dataset['learning-step'], 
            self.aggregated_dataset['reward-mean'], 
            yerr=self.aggregated_dataset['reward-std'], 
            label='Mean [std]',
            color='blue', marker='o', linestyle='solid', linewidth=2, markersize=8, capsize=5)
        axs[0].plot(
            self.aggregated_dataset['learning-step'], 
            self.aggregated_dataset['reward-min'], 
            label='Min', color='red')
        axs[0].plot(
            self.aggregated_dataset['learning-step'], 
            self.aggregated_dataset['reward-max'], 
            label='Max', color='green')
        axs[0].plot(
            self.aggregated_dataset['learning-step'], 
            self.aggregated_dataset['reward-median'], 
            label='Median', color='cyan')
        axs[0].set_ylabel('Reward')
        axs[0].set_ylim(-20000, 0)
        axs[0].grid(True)
        axs[0].legend(loc=1, ncol=4, shadow=True)

        axs[1].errorbar(
            self.aggregated_dataset['learning-step'], 
            self.aggregated_dataset['reward-mean'], 
            yerr=self.aggregated_dataset['reward-std'], 
            label='Mean [std]',
            color='blue', marker='o', linestyle='solid', linewidth=2, markersize=8, capsize=5)
        axs[1].plot(
            self.aggregated_dataset['learning-step'], 
            self.aggregated_dataset['reward-min'], 
            label='Min', color='red')
        axs[1].plot(
            self.aggregated_dataset['learning-step'], 
            self.aggregated_dataset['reward-max'], 
            label='Max', color='green')
        axs[1].plot(
            self.aggregated_dataset['learning-step'], 
            self.aggregated_dataset['reward-median'], 
            label='Median', color='cyan')
        axs[1].set_ylabel('Reward')
        axs[1].set_xlabel('Learning step')
        axs[1].grid(True)

        axs[2].plot(
            self.aggregated_dataset['learning-step'], 
            self.aggregated_dataset['percentage-waited'], 
            label='% Waiting too long', color='black', linestyle='dotted')
        axs[2].plot(
            self.aggregated_dataset['learning-step'], 
            self.aggregated_dataset['percentage-stable'], 
            label='% Stable policy', color='magenta', linestyle='dashed')
        axs[2].set_ylabel('Agents [%]')
        axs[2].set_ylim(-10, 110)
        axs[2].set_xlabel('Learning step')
        axs[2].grid(True)
        axs[2].legend(loc=1, ncol=2, shadow=True)

        print('Saving plots...')
        fig.savefig('{}.svg'.format(self.output_prefix),
                    dpi=300, transparent=False, bbox_inches='tight')
        # fig.savefig('{}.png'.format(self.output_prefix),
        #             dpi=300, transparent=False, bbox_inches='tight')
        # plt.show()   
        matplotlib.pyplot.close('all')

    def generate_plot_old(self):
        """ Plots the aggregated data. """
        fig, host = plt.subplots(figsize=(15, 10))
        perc = host.twinx()
        p1 = host.errorbar(
            self.aggregated_dataset['learning-step'], 
            self.aggregated_dataset['reward-mean'], 
            yerr=self.aggregated_dataset['reward-std'], 
            label='Mean [std]',
            color='blue', marker='o', linestyle='solid', linewidth=2, markersize=8, capsize=5)
        p2 = host.plot(
            self.aggregated_dataset['learning-step'], 
            self.aggregated_dataset['reward-min'], 
            label='Min', color='red')
        p3 = host.plot(
            self.aggregated_dataset['learning-step'], 
            self.aggregated_dataset['reward-max'], 
            label='Max', color='green')
        p4 = host.plot(
            self.aggregated_dataset['learning-step'], 
            self.aggregated_dataset['reward-median'], 
            label='Median', color='cyan')
        p5 = perc.plot(
            self.aggregated_dataset['learning-step'], 
            self.aggregated_dataset['percentage-waited'], 
            label='% Waiting too long', color='orange', linestyle='dotted')
        p6 = perc.plot(
            self.aggregated_dataset['learning-step'], 
            self.aggregated_dataset['percentage-stable'], 
            label='% Stable policy', color='magenta', linestyle='dashed')

        host.set_xlabel('Learning step')

        host.set_ylabel('Reward')
        # host.set_ylim(-86400.0, 0.0)

        perc.set_ylabel('Agents [%]')
        perc.set_ylim(-10, 110)

        plots = [p1 , p2[0], p3[0], p4[0], p5[0], p6[0]]
        host.legend(plots, [l.get_label() for l in plots], shadow=True,
                    bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                    ncol=3, mode="expand", borderaxespad=0.)
        host.grid()
        fig.savefig('{}.svg'.format(self.output_prefix),
                    dpi=300, transparent=False, bbox_inches='tight')
        # fig.savefig('{}.png'.format(self.output_prefix),
        #             dpi=300, transparent=False, bbox_inches='tight')
        # plt.show()   
        matplotlib.pyplot.close('all')

    ####################################### GENERIC  GETTERS #######################################

    def _agent_waited_too_long(self, agent):
        if agent not in self.aggregated_dataset['waited']:
            # covers the initial empty structure
            return False
        return self.aggregated_dataset['waited'][agent] is not None
    
    ################################################################################################

####################################################################################################

if __name__ == '__main__':
    _main()

####################################################################################################