#!/usr/bin/env python3

""" Process the DBLogger directory structure plotting the agents decision making outcome. """

import argparse
from collections import defaultdict
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
        '--profiler', dest='profiler', action='store_true', help='Enable cProfile.')
    parser.set_defaults(profiler=False)
    return parser.parse_args()

def _main():
    """ Process the DBLogger directory structure plotting the agents decision making outcome. """

    config = _argument_parser()

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler = cProfile.Profile()
        profiler.enable()
    ## ========================              PROFILER              ======================== ##

    statistics = AggregatedAgentsOutcome(config.dir_tree, config.data, config.graph)
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

class AggregatedAgentsOutcome(DBLoggerStats):
    """ Process the DBLogger directory structure plotting the agents decision making outcome. """

    MODES = { 
        '0': 'wait too long',
        '1': 'passenger',
        '2': 'public',
        '3': 'walk',
        '4': 'bicycle',
        '5': 'ptw',
        '6': 'on-demand',
    }

    def __init__(self, directory, dataset, prefix):
        super().__init__(directory)
        self.dataset_fname = dataset
        self.output_prefix = prefix
        self.aggregated_dataset = dict()

    ####################################### DATA  AGGREGATOR #######################################

    def _init_datastructure(self):
        """ Loads the dataset from file if exist or generates the empty structure. """

        if os.path.exists(self.dataset_fname):
            with open(self.dataset_fname, 'r') as jsonfile:
                self.aggregated_dataset = json.load(jsonfile)
        else: 
            # Y - average
            self.aggregated_dataset['modes-departure'] = { 
                '1': list(),
                '2': list(),
                '3': list(),
                '4': list(),
                '5': list(),
                '6': list(),
            }
            # Y - average
            self.aggregated_dataset['modes-waiting'] = { 
                '1': list(),
                '2': list(),
                '3': list(),
                '4': list(),
                '5': list(),
                '6': list(),
            }
            # Y - number of agents
            self.aggregated_dataset['modes-late'] = { 
                '1': list(),
                '2': list(),
                '3': list(),
                '4': list(),
                '5': list(),
                '6': list(),
            }
            # Y - number of agents
            self.aggregated_dataset['waiting-too-long'] = list()
            # Y - number of agents
            self.aggregated_dataset['mistake'] = list()
            # X
            self.aggregated_dataset['episodes'] = list()
            # min / max y
            self.aggregated_dataset['min-agents'] = float('inf')
            self.aggregated_dataset['max-agents'] = float('-inf')
            self.aggregated_dataset['min-departure'] = float('inf')
            self.aggregated_dataset['max-departure'] = float('-inf')
            self.aggregated_dataset['min-wait'] = float('inf')
            self.aggregated_dataset['max-wait'] = float('-inf')
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
            agents, episodes, _ = self.get_training_components(training_run)
            
            # process the info file
            for episode in episodes:
                self.aggregated_dataset['episodes'].append(
                    len(self.aggregated_dataset['episodes']) + 1)
                temp = defaultdict(lambda: defaultdict(lambda: list()))
                mistake = 0
                too_long = 0
                for agent in agents:
                    info = self.get_info(training_run, episode, agent)
                    # {'arrival': 30629.0,
                    #  'cost': 1611.5221600000002,
                    #  'departure': 28787.0,
                    #  'discretized-cost': 13,
                    #  'ett': 1611.5221600000186,
                    #  'ext': {'bicycle': [559.8150000000002],
                    #          'on-demand': [235.57260897450837],
                    #          'passenger': [235.57260897450837],
                    #          'ptw': [235.57260897450837],
                    #          'public': [1611.5221600000002],
                    #          'walk': [2884.6463999999996]},
                    #  'from-state': {'ett': '[ 5  2  2  2 13 24]',
                    #                 'from': 213,
                    #                 'time-left': 30,
                    #                 'to': 331,
                    #                 'usage': [0, 0, 0, 0, 2, 0]},
                    #  'mode': 'public',
                    #  'rtt': 1842.0,
                    #  'timeLoss': 235.33,
                    #  'wait': 1771.0}
                    last_action = str(self.get_last_action(training_run, episode, agent))
                    ### CASES
                    if last_action != '0' and info['mode'] is None:
                        ### MISTAKE
                        mistake += 1
                    elif last_action == '0':
                        ### WAITED TOO LONG
                        too_long += 1
                    else:
                        temp[last_action]['departure'].append(info['departure'])
                        if info['wait'] <= 0:
                            ### TOO LATE
                            temp[last_action]['late'].append(info['wait'])
                        else:
                            temp[last_action]['wait'].append(info['wait'])
                # aggregation
                for mode in self.aggregated_dataset['modes-departure']:
                    mean = np.mean(temp[mode]['departure']) / 3600.0
                    self.aggregated_dataset['modes-departure'][mode].append(mean)
                    if not np.isnan(mean):
                        self.aggregated_dataset['min-departure'] = min(
                            self.aggregated_dataset['min-departure'], mean)
                        self.aggregated_dataset['max-departure'] = max(
                            self.aggregated_dataset['max-departure'], mean)
                for mode in self.aggregated_dataset['modes-waiting']:
                    mean = np.mean(temp[mode]['wait']) / 60.0
                    self.aggregated_dataset['modes-waiting'][mode].append(mean)
                    if not np.isnan(mean):
                        self.aggregated_dataset['min-wait'] = min(
                            self.aggregated_dataset['min-wait'], mean)
                        self.aggregated_dataset['max-wait'] = max(
                            self.aggregated_dataset['max-wait'], mean)
                for mode in self.aggregated_dataset['modes-late']:
                    self.aggregated_dataset['modes-late'][mode].append(len(temp[mode]['late']))
                    self.aggregated_dataset['min-agents'] = min(
                        self.aggregated_dataset['min-agents'], len(temp[mode]['late']))
                    self.aggregated_dataset['max-agents'] = max(
                        self.aggregated_dataset['max-agents'], len(temp[mode]['late']))

                self.aggregated_dataset['mistake'].append(mistake)
                self.aggregated_dataset['min-agents'] = min(
                        self.aggregated_dataset['min-agents'], mistake)
                self.aggregated_dataset['max-agents'] = max(
                        self.aggregated_dataset['max-agents'], mistake)
                self.aggregated_dataset['waiting-too-long'].append(too_long)
                self.aggregated_dataset['min-agents'] = min(
                        self.aggregated_dataset['min-agents'], too_long)
                self.aggregated_dataset['max-agents'] = max(
                        self.aggregated_dataset['max-agents'], too_long)

            self.aggregated_dataset['training-folders'].append(training_run)

        LOGGER.debug('UPDATED aggregated data structure: \n%s', pformat(self.aggregated_dataset))

        # save the new dataset into the dataset file
        self._save_satastructure() 

    ######################################## PLOT GENERATOR ########################################

    def generate_plot(self):
        print('Plotting..')
        fig, axs = plt.subplots(
            len(self.MODES), 3, sharex=True, figsize=(20, 25), constrained_layout=True)
        
        delta_agents = self.aggregated_dataset['max-agents'] - self.aggregated_dataset['min-agents']
        delta_agents *= 0.1 
        min_agents = self.aggregated_dataset['min-agents'] - delta_agents
        max_agents = self.aggregated_dataset['max-agents'] + delta_agents

        delta_departure = self.aggregated_dataset['max-departure'] - self.aggregated_dataset['min-departure']
        delta_departure *= 0.1 
        min_departure = self.aggregated_dataset['min-departure'] - delta_departure
        max_departure = self.aggregated_dataset['max-departure'] + delta_departure

        delta_wait = self.aggregated_dataset['max-wait'] - self.aggregated_dataset['min-wait']
        delta_wait *= 0.1 
        min_wait = self.aggregated_dataset['min-wait'] - delta_wait
        max_wait = self.aggregated_dataset['max-wait'] + delta_wait

        # generic plots mode independent
        axs[0][0].set_title('Mistakes')
        axs[0][0].plot(
            self.aggregated_dataset['episodes'],
            self.aggregated_dataset['mistake'],
            label='Mistakes')
        axs[0][0].set_ylabel('Agents [#]')
        axs[0][0].set_ylim(min_agents, max_agents)
        axs[0][0].grid(True)
        # axs[0][0].legend(loc=1, ncol=1, shadow=True)

        axs[0][1].set_title('Empty Plot :)')

        axs[0][2].set_title('Waiting too long')
        axs[0][2].plot(
            self.aggregated_dataset['episodes'],
            self.aggregated_dataset['waiting-too-long'],
            label='Waiting too long')
        axs[0][2].set_ylabel('Agents [#]')
        axs[0][2].set_ylim(min_agents, max_agents)
        axs[0][2].grid(True)
        # axs[0][2].legend(loc=1, ncol=1, shadow=True)

        for mode in sorted(self.MODES):
            if mode == '0':
                continue
            axs[int(mode)][0].set_title('{} - Departure'.format(self.MODES[mode]))
            axs[int(mode)][0].plot(
                self.aggregated_dataset['episodes'], 
                self.aggregated_dataset['modes-departure'][mode], 
                label='Average departure')
            axs[int(mode)][0].set_ylabel('Time [h]')
            axs[int(mode)][0].set_ylim(min_departure, max_departure)
            axs[int(mode)][0].grid(True)
            # axs[int(mode)][0].legend(loc=1, ncol=1, shadow=True)

            axs[int(mode)][1].set_title('{} - Waiting'.format(self.MODES[mode]))
            axs[int(mode)][1].plot(
                self.aggregated_dataset['episodes'], 
                self.aggregated_dataset['modes-waiting'][mode], 
                label='Average waiting')
            axs[int(mode)][1].set_ylabel('Time [m]')
            axs[int(mode)][1].set_ylim(min_wait, max_wait)
            axs[int(mode)][1].grid(True)
            # axs[int(mode)][1].legend(loc=1, ncol=1, shadow=True)

            axs[int(mode)][2].set_title('{} - Too late'.format(self.MODES[mode]))
            axs[int(mode)][2].plot(
                self.aggregated_dataset['episodes'], 
                self.aggregated_dataset['modes-late'][mode], 
                label='Late arrival')
            axs[int(mode)][2].set_ylabel('Agents [#]')
            axs[int(mode)][2].set_ylim(min_agents, max_agents)
            axs[int(mode)][2].grid(True)
            # axs[int(mode)][2].legend(loc=1, ncol=1, shadow=True)

        axs[len(self.MODES)-1][0].set_xlabel('Episodes')
        axs[len(self.MODES)-1][1].set_xlabel('Episodes')        
        axs[len(self.MODES)-1][2].set_xlabel('Episodes')

        print('Saving to file..')
        fig.savefig('{}.svg'.format(self.output_prefix),
                    dpi=300, transparent=False, bbox_inches='tight')
        fig.savefig('{}.png'.format(self.output_prefix),
                    dpi=300, transparent=False, bbox_inches='tight')
        # plt.show()   
        matplotlib.pyplot.close('all')   

####################################################################################################

if __name__ == '__main__':
    _main()

####################################################################################################