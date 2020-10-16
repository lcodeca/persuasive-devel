#!/usr/bin/env python3

""" Process the RLLIB logs/result.json """

import argparse
import collections
import cProfile
import io
import json
import logging
import os
import pstats
import random
import sys
from pprint import pformat, pprint

import matplotlib
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as geometry
from tqdm import tqdm

# """ Import SUMO library """
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
    import sumolib
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

####################################################################################################

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

random.seed()

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
    parser = argparse.ArgumentParser(description='RLLIB & SUMO Statistics parser.')
    parser.add_argument('--results', required=True, type=str, help='RESULTS, JSON file.')
    parser.add_argument('--init', required=True, type=str, help='AGENT INIT, JSON file.')
    parser.add_argument('--prefix', default='stats', help='Output prefix for the processed data.')
    parser.add_argument('--max', default=None, type=int, help='Maximum number of runs.')
    parser.add_argument('--profiler', dest='profiler', action='store_true', help='Enable cProfile.')
    parser.set_defaults(profiler=False)
    return parser.parse_args()

class Policy(object):
    """ Process a SINGLE RLLIB logs/result.json file as a time series. """

    def __init__(self, agents, results, prefix, max_runs):
        with open(agents, 'r') as jsonfile:
            self.agents_init = json.load(jsonfile)
        self.max_runs = max_runs
        self.results_file = results
        self.prefix = prefix
        self.agent_learning = collections.defaultdict(lambda: collections.defaultdict(dict))
        self.agent_evaluation = collections.defaultdict(lambda: collections.defaultdict(dict))

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

    def historical_policies(self):
        counter = 0
        labels = [
            # mpatches.Patch(color='black', label='Wait'),
            mpatches.Patch(color='red', label='Car'),
            mpatches.Patch(color='green', label='PT'),
            mpatches.Patch(color='orange', label='Walk'),
            mpatches.Patch(color='blue', label='Bicycle'),
            mpatches.Patch(color='purple', label='PTW'),
        ]

        modes_color = {
            0: 'black',
            1: 'red',
            2: 'green',
            3: 'orange',
            4: 'blue',
            5: 'purple',
        }

        with open(self.results_file, 'r') as jsonfile:
            for row in tqdm(jsonfile): # enumerate cannot be used due to the size of the file

                ################################################################
                ##                     Setup the images
                ################################################################

                fig, axs = plt.subplots(1, 2, figsize=(25, 15), constrained_layout=True)
                fig.suptitle('Policy for training run {}'.format(counter))

                ################################################################

                complete = json.loads(row)

                ############ LEARNIN
                axs[0].bar(9.0, len(self.agents_init), width=0.05, color='r', align='center')
                axs[0].set_xlim(6.0, 10.0)
                axs[0].set_ylim(-1, len(self.agents_init))
                axs[0].set_title('Learning')
                axs[0].set_xlabel('Time [h]')
                axs[0].set_ylabel('Agents [#]')
                axs[0].legend(handles=labels, loc='upper left')

                info_by_episode = complete['hist_stats']['info_by_agent']
                last_action_by_agent = complete['hist_stats']['last_action_by_agent']
                # rewards_by_agent = complete['hist_stats']['rewards_by_agent']
                pos = random.randrange(len(info_by_episode))
                episode = info_by_episode[pos]
                for agent, info in episode.items():
                    try:
                        y_agent = int(agent.split('_')[1])
                        start = self.agents_init[agent]['start']/3600
                        mode = last_action_by_agent[pos][agent]
                        arrival = info['arrival']/3600.0
                        departure = info['departure']/3600.0

                        y = [y_agent, y_agent]
                        axs[0].plot([start, departure], y, color='black', alpha=0.75)
                        axs[0].plot([departure, arrival], y, color=modes_color[mode], alpha=0.75)
                    except KeyError as exception:
                        print(exception)
                        logger.critical('[Learning] Missing stats in %d', counter)

                ############ EVALUATION
                axs[1].bar(9.0, len(self.agents_init), width=0.05, color='r', align='center')
                axs[1].set_xlim(6.0, 10.0)
                axs[1].set_ylim(-1, len(self.agents_init))
                axs[1].set_title('Evaluation')
                axs[1].set_xlabel('Time [h]')
                axs[1].set_ylabel('Agents [#]')
                axs[1].legend(handles=labels, loc='upper left')

                if 'evaluation' in complete:
                    complete = complete['evaluation']
                    info_by_episode = complete['hist_stats']['info_by_agent']
                    last_action_by_agent = complete['hist_stats']['last_action_by_agent']
                    # rewards_by_agent = complete['hist_stats']['rewards_by_agent']
                    pos = random.randrange(len(info_by_episode))
                    episode = info_by_episode[pos]
                    for agent, info in episode.items():
                        try:
                            y_agent = int(agent.split('_')[1])
                            start = self.agents_init[agent]['start']/3600
                            mode = last_action_by_agent[pos][agent]
                            arrival = info['arrival']/3600.0
                            departure = info['departure']/3600.0

                            y = [y_agent, y_agent]
                            axs[1].plot([start, departure], y, color='black', alpha=0.99)
                            axs[1].plot([departure, arrival], y, color=modes_color[mode], alpha=0.99)
                        except KeyError as exception:
                            print(exception)
                            logger.critical('[Evaluation] Missing stats in %d', counter)

                ################################################################
                fig.savefig('{}.policy.{}.svg'.format(self.prefix, counter),
                            dpi=300, transparent=False, bbox_inches='tight')
                # fig.savefig('{}.{}.png'.format(self.prefix, self.agent_name),
                #             dpi=300, transparent=False, bbox_inches='tight')
                # plt.show()
                matplotlib.pyplot.close('all')
                # sys.exit()
                ###############################################################
                counter += 1
                if self.max_runs is not None:
                    if counter > self.max_runs:
                        break

        #################################

def _main():
    """ Process the RLLIB logs/result.json """

    config = _argument_parser()

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler = cProfile.Profile()
        profiler.enable()
    ## ========================              PROFILER              ======================== ##

    statistics = Policy(config.init, config.results, config.prefix, config.max)
    statistics.historical_policies()
    logging.info('Done')

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler.disable()
        results = io.StringIO()
        pstats.Stats(profiler, stream=results).sort_stats('cumulative').print_stats(50)
        logging.info('Profiler: \n%s', pformat(results.getvalue()))
    ## ========================              PROFILER              ======================== ##

if __name__ == '__main__':
    _main()
