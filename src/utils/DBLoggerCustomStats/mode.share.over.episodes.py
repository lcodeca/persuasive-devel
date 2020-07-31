#!/usr/bin/env python3

"""
    Process the DBLogger directory structure generating an overview
    of the mode share over the episodes.
"""

import argparse
import cProfile
import io
import json
import logging
import os
from pprint import pformat
import pstats

import matplotlib
import matplotlib.pyplot as plt
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
    """
        Process the DBLogger directory structure generating an overview
        of the mode share over the episodes.
    """

    config = _argument_parser()

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler = cProfile.Profile()
        profiler.enable()
    ## ========================              PROFILER              ======================== ##

    statistics = ModeShareEpisodes(config.dir_tree, config.data, config.graph)
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

class ModeShareEpisodes(DBLoggerStats):
    """
        Process the DBLogger directory structure generating an overview
        of the mode share over the episodes.
    """
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
            # mode share over episodes
            self.aggregated_dataset['mode'] = {
                '0': list(),
                '1': list(),
                '2': list(),
                '3': list(),
                '4': list(),
                '5': list(),
                '6': list(),
            }
            self.aggregated_dataset['episodes'] = list()
            self.aggregated_dataset['ylim'] = 0

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
        for training_run in tqdm(available_training_runs):
            if training_run in self.aggregated_dataset['training-folders']:
                continue
            print('Processing {}/{}'.format(self.dir, training_run))
            agents, episodes, _ = self.get_training_components(training_run)

            # process the info file
            for episode in episodes:
                self.aggregated_dataset['episodes'].append(
                    len(self.aggregated_dataset['episodes']) + 1)
                modes = {
                    '0': 0,
                    '1': 0,
                    '2': 0,
                    '3': 0,
                    '4': 0,
                    '5': 0,
                    '6': 0,
                }
                for agent in agents:
                    action = self.get_last_action(training_run, episode, agent)
                    modes[str(action)] += 1
                for mode, counter in modes.items():
                    self.aggregated_dataset['mode'][mode].append(counter)
                    self.aggregated_dataset['ylim'] = max(self.aggregated_dataset['ylim'], counter)

            self.aggregated_dataset['training-folders'].append(training_run)

        LOGGER.debug('UPDATED aggregated data structure: \n%s', pformat(self.aggregated_dataset))

        # save the new dataset into the dataset file
        self._save_satastructure()

    ######################################## PLOT GENERATOR ########################################

    def generate_plot(self):
        """ Plots the aggregated data. """
        fig, axs = plt.subplots(
            len(self.MODES), sharex=True, figsize=(15, 25), constrained_layout=True)

        for mode in sorted(self.MODES):
            axs[int(mode)].plot(
                self.aggregated_dataset['episodes'], self.aggregated_dataset['mode'][mode],
                label=self.MODES[mode])
            axs[int(mode)].set_ylabel('Agents [#]')
            axs[int(mode)].set_ylim(0, self.aggregated_dataset['ylim'])
            axs[int(mode)].grid(True)
            axs[int(mode)].legend(loc=1, ncol=1, shadow=True)
        axs[len(self.MODES)-1].set_xlabel('Episodes')
        fig.savefig('{}.svg'.format(self.output_prefix),
                    dpi=300, transparent=False, bbox_inches='tight')
        # fig.savefig('{}.png'.format(self.output_prefix),
        #             dpi=300, transparent=False, bbox_inches='tight')
        # plt.show()
        matplotlib.pyplot.close('all')

    def generate_messy_plot(self):
        """ Plots the aggregated data. """

        fig, main = plt.subplots(figsize=(15, 10))

        main.plot(self.aggregated_dataset['episodes'], self.aggregated_dataset['mode']['0'],
                  label=self.MODES['0'])
        for mode in sorted(self.aggregated_dataset['mode'].keys()):
            if mode == '0':
                # already done.
                continue
            main.plot(self.aggregated_dataset['episodes'], self.aggregated_dataset['mode'][mode],
                      label=self.MODES[mode])

        main.set_xlabel('Episodes')
        main.set_ylabel('Agents [#]')

        main.legend(shadow=True, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                    ncol=round(len(self.aggregated_dataset['mode'])/2.0),
                    mode="expand", borderaxespad=0.)
        main.grid()
        fig.savefig('{}.svg'.format(self.output_prefix),
                    dpi=300, transparent=False, bbox_inches='tight')
        # fig.savefig('{}.png'.format(self.output_prefix),
        #             dpi=300, transparent=False, bbox_inches='tight')
        plt.show()
        matplotlib.pyplot.close('all')

    def generate_bar_plot(self):
        """ Plots the aggregated data. """

        fig, main = plt.subplots(figsize=(15, 10))

        main.bar(self.aggregated_dataset['episodes'], self.aggregated_dataset['mode']['0'],
                 0.5, label=self.MODES['0'])
        bottom_graph = self.aggregated_dataset['mode']['0']
        for mode in sorted(self.aggregated_dataset['mode'].keys()):
            if mode == '0':
                # already done.
                continue
            main.bar(self.aggregated_dataset['episodes'], self.aggregated_dataset['mode'][mode],
                     0.5, label=self.MODES[mode], bottom=bottom_graph)
            bottom_graph = self.aggregated_dataset['mode'][mode]

        main.set_xlabel('Episodes')
        main.set_ylabel('Agents [#]')

        main.legend(shadow=True, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                    ncol=round(len(self.aggregated_dataset['mode'])/2.0),
                    mode="expand", borderaxespad=0.)
        main.grid()
        fig.savefig('{}.svg'.format(self.output_prefix),
                    dpi=300, transparent=False, bbox_inches='tight')
        # fig.savefig('{}.png'.format(self.output_prefix),
        #             dpi=300, transparent=False, bbox_inches='tight')
        plt.show()
        matplotlib.pyplot.close('all')

####################################################################################################

if __name__ == '__main__':
    _main()

####################################################################################################
