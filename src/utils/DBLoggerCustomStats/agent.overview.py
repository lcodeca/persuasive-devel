#!/usr/bin/env python3

""" Process the DBLogger directory structure generating an agents overview. """

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
    """ Process the DBLogger directory structure generating an agents overview. """

    config = _argument_parser()

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler = cProfile.Profile()
        profiler.enable()
    ## ========================              PROFILER              ======================== ##

    statistics = AgentOverview(config.dir_tree, config.data, config.graph)
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

class AgentOverview(DBLoggerStats):

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
            # Agent overview from info.json
            self.aggregated_dataset['agents'] = defaultdict(lambda: defaultdict(lambda: list()))
            self.aggregated_dataset['episodes'] = list()

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
            
            # process the info file
            for episode in episodes:
                self.aggregated_dataset['episodes'].append(
                    len(self.aggregated_dataset['episodes']) + 1)
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
                    self.aggregated_dataset['agents'][agent]['arrival'].append(info['arrival']/3600.0)
                    self.aggregated_dataset['agents'][agent]['cost'].append(info['cost']/60.0)
                    self.aggregated_dataset['agents'][agent]['departure'].append(info['departure']/3600.0)
                    self.aggregated_dataset['agents'][agent]['ett'].append(info['ett']/60.0)
                    self.aggregated_dataset['agents'][agent]['rtt'].append(info['rtt']/60.0)
                    self.aggregated_dataset['agents'][agent]['timeLoss'].append(info['timeLoss']/60.0)
                    self.aggregated_dataset['agents'][agent]['wait'].append(info['wait']/60.0)
                    self.aggregated_dataset['agents'][agent]['difference'].append((info['ett'] - info['rtt'])/60.0)
                    
                    sequence = self.get_learning_sequence(training_run, episode, agent)
                    self.aggregated_dataset['agents'][agent]['reward'].append(sequence[-1][3])
                    self.aggregated_dataset['agents'][agent]['mode'].append(sequence[-1][1])
                    self.aggregated_dataset['agents'][agent]['actions'].append(len(sequence))

            self.aggregated_dataset['training-folders'].append(training_run)

        LOGGER.debug('UPDATED aggregated data structure: \n%s', pformat(self.aggregated_dataset))

        # save the new dataset into the dataset file
        self._save_satastructure() 

    ######################################## PLOT GENERATOR ########################################

    def generate_plot(self):
        print('Plotting agents..')
        for agent, stats in tqdm(self.aggregated_dataset['agents'].items()):
            # https://matplotlib.org/gallery/subplots_axes_and_figures/ganged_plots.html#sphx-glr-gallery-subplots-axes-and-figures-ganged-plots-py
            fig, axs = plt.subplots(5, 2, sharex=True, figsize=(20, 20), constrained_layout=True)
            fig.suptitle('{}'.format(agent))

            ett_rtt_max = max(max(stats['ett']),max(stats['rtt']))
            ett_rtt_max += ett_rtt_max * 0.1

            # Plot each graph
            axs[0][0].plot(self.aggregated_dataset['episodes'], stats['reward'], label='Reward',
                           color='blue', marker='o', linestyle='solid', linewidth=2, markersize=8)
            axs[0][0].set_ylabel('Reward')
            axs[0][0].grid(True)

            axs[1][0].plot(self.aggregated_dataset['episodes'], stats['actions'], label='Number of actions',
                           color='red', marker='o', linestyle='solid', linewidth=2, markersize=8)
            axs[1][0].set_ylabel('Actions [#]')
            axs[1][0].grid(True)
            
            axs[2][0].plot(self.aggregated_dataset['episodes'], stats['mode'], label='Selected mode',
                           color='green', marker='o', linestyle='solid', linewidth=2, markersize=8)
            axs[4][0].set_ylim(0, max(stats['mode']))
            axs[2][0].set_ylabel('Mode')
            axs[2][0].grid(True)
            
            axs[3][0].plot(self.aggregated_dataset['episodes'], stats['ett'], label='Estimated Travel Time',
                           color='black', marker='o', linestyle='solid', linewidth=2, markersize=8)
            axs[3][0].set_ylim(0, ett_rtt_max)
            axs[3][0].set_ylabel('Est TT [m]')
            axs[3][0].grid(True)
            
            axs[4][0].plot(self.aggregated_dataset['episodes'], stats['rtt'], label='Real Travel Time',
                           color='magenta', marker='o', linestyle='solid', linewidth=2, markersize=8)
            axs[4][0].set_ylim(0, ett_rtt_max)
            axs[4][0].set_ylabel('Real TT [m]')
            axs[4][0].set_xlabel('Episode [#]')
            axs[4][0].grid(True)

            axs[0][1].plot(self.aggregated_dataset['episodes'], stats['departure'], 'b-', label='Departure',
                           color='blue', marker='o', linestyle='solid', linewidth=2, markersize=8)
            axs[0][1].axhline(y=9.0, color='red', linestyle='dashed')
            axs[0][1].set_ylabel('Departure [h]')
            axs[0][1].grid(True)
            
            axs[1][1].plot(self.aggregated_dataset['episodes'], stats['arrival'], 'r-', label='Arrival',
                           color='red', marker='o', linestyle='solid', linewidth=2, markersize=8)
            axs[1][1].axhline(y=9.0, color='red', linestyle='dashed')
            axs[1][1].set_ylabel('Arrival [h]')
            axs[1][1].grid(True)

            axs[2][1].plot(self.aggregated_dataset['episodes'], stats['wait'], 'g-', label='Waiting at destination',
                           color='green', marker='o', linestyle='solid', linewidth=2, markersize=8)
            axs[2][1].axhline(y=0.0, color='red', linestyle='dashed')
            axs[2][1].set_ylabel('Wait @ destination [m]')
            axs[2][1].grid(True)

            axs[3][1].plot(self.aggregated_dataset['episodes'], stats['cost'], 'k-', label='Estimated cost',
                           color='black', marker='o', linestyle='solid', linewidth=2, markersize=8)
            axs[3][1].set_ylabel('Est Cost [m]')
            axs[3][1].grid(True)

            axs[4][1].plot(self.aggregated_dataset['episodes'], stats['difference'], 'm-', label='ETT / RTT Difference',
                           color='magenta', marker='o', linestyle='solid', linewidth=2, markersize=8)
            axs[4][1].axhline(y=0.0, color='red', linestyle='dashed')
            axs[4][1].set_ylabel('ETT / RTT Difference [m]')
            axs[4][1].set_xlabel('Episode [#]')
            axs[4][1].grid(True)

            fig.savefig('{}.{}.svg'.format(self.output_prefix, agent),
                    dpi=300, transparent=False, bbox_inches='tight')
            # fig.savefig('{}.{}.png'.format(self.output_prefix, agent),
            #         dpi=300, transparent=False, bbox_inches='tight')
            # plt.show()
            matplotlib.pyplot.close('all')     

####################################################################################################

if __name__ == '__main__':
    _main()

####################################################################################################