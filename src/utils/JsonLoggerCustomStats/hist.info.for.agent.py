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
    parser.add_argument('--agent', default='agent_0', type=str, help='Agent name.')
    parser.add_argument('--shapes', required=True, type=str, help='The SUMO shapefile with the TAZ shape.')
    parser.add_argument('--image', required=True, type=str, help='The SUMO NET Image.png.')
    parser.add_argument('--max', default=None, type=int, help='Maximum number of runs.')
    parser.add_argument('--all', dest='all_agents', action='store_true', help='Enable cProfile.')
    parser.set_defaults(all_agents=False)
    parser.add_argument('--profiler', dest='profiler', action='store_true', help='Enable cProfile.')
    parser.set_defaults(profiler=False)
    return parser.parse_args()

class AgentStats(object):
    """ Process a SINGLE RLLIB logs/result.json file as a time series. """

    def __init__(self, agents, results, prefix, agent_name, shapes, all_agents, max_runs, image):
        self.all_agents = all_agents
        with open(agents, 'r') as jsonfile:
            self.agents_init = json.load(jsonfile)
        self.agent_name = agent_name
        self.agent = self.agents_init[agent_name]
        self.max_runs = max_runs

        self.results_file = results
        self.prefix = prefix
        self.agent_learning = collections.defaultdict(lambda: collections.defaultdict(dict))
        self.agent_evaluation = collections.defaultdict(lambda: collections.defaultdict(dict))

        self.shape = None
        for poly in sumolib.shapes.polygon.read(shapes, includeTaz=True):
            if poly.id == 'taz':
                self.shape = geometry.MultiPoint(poly.shape).convex_hull
                break

        self.image = mpimg.imread(image)

    @staticmethod
    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    def _load_shapefile(self):
        """ Load SUMO net.xml and extract the edges """

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

    def historical_info_all_agents(self):
        for agent, init in tqdm(self.agents_init.items()):
            self.agent_name = agent
            self.agent = init
            self.historical_info_single_agent()

    def historical_info_single_agent(self):
        logger.info('Computing the detailed info for agent %s over the episodes.', self.agent_name)
        counter = 0
        with open(self.results_file, 'r') as jsonfile:
            for row in tqdm(jsonfile): # enumerate cannot be used due to the size of the file
                complete = json.loads(row)
                #### LEARNING
                info_by_episode = complete['hist_stats']['info_by_agent']
                last_action_by_agent = complete['hist_stats']['last_action_by_agent']
                rewards_by_agent = complete['hist_stats']['rewards_by_agent']
                for pos, episode in enumerate(info_by_episode):
                    try:
                        info = episode[self.agent_name]
                        #############
                        self.agent_learning[counter][pos]['reward'] = sum(rewards_by_agent[pos][self.agent_name])
                        self.agent_learning[counter][pos]['actions'] = len(rewards_by_agent[pos][self.agent_name])
                        self.agent_learning[counter][pos]['mode'] = last_action_by_agent[pos][self.agent_name]
                        #############
                        self.agent_learning[counter][pos]['cost'] = info['cost']/60.0
                        self.agent_learning[counter][pos]['discretized-cost'] = info['discretized-cost']
                        self.agent_learning[counter][pos]['rtt'] = info['rtt']/60.0
                        #############
                        self.agent_learning[counter][pos]['arrival'] = info['arrival']/3600.0
                        self.agent_learning[counter][pos]['departure'] = info['departure']/3600.0
                        self.agent_learning[counter][pos]['ett'] = info['ett']/60.0
                        self.agent_learning[counter][pos]['wait'] = info['wait']/60.0
                    except KeyError as exception:
                        print(exception)
                        logger.critical('[Learning] Missing stats in %d - %d', counter, pos)

                #### EVALUATION
                if 'evaluation' in complete:
                    complete = complete['evaluation']
                    info_by_episode = complete['hist_stats']['info_by_agent']
                    last_action_by_agent = complete['hist_stats']['last_action_by_agent']
                    rewards_by_agent = complete['hist_stats']['rewards_by_agent']
                    for pos, episode in enumerate(info_by_episode):
                        try:
                            info = episode[self.agent_name]
                            #############
                            self.agent_evaluation[counter][pos]['reward'] = sum(rewards_by_agent[pos][self.agent_name])
                            self.agent_evaluation[counter][pos]['actions'] = len(rewards_by_agent[pos][self.agent_name])
                            self.agent_evaluation[counter][pos]['mode'] = last_action_by_agent[pos][self.agent_name]
                            #############
                            self.agent_evaluation[counter][pos]['cost'] = info['cost']/60.0
                            self.agent_evaluation[counter][pos]['discretized-cost'] = info['discretized-cost']
                            self.agent_evaluation[counter][pos]['rtt'] = info['rtt']/60.0
                            #############
                            self.agent_evaluation[counter][pos]['arrival'] = info['arrival']/3600.0
                            self.agent_evaluation[counter][pos]['departure'] = info['departure']/3600.0
                            self.agent_evaluation[counter][pos]['ett'] = info['ett']/60.0
                            self.agent_evaluation[counter][pos]['wait'] = info['wait']/60.0
                        except KeyError as exception:
                            print(exception)
                            logger.critical('[Evaluation] Missing stats in %d - %d', counter, pos)
                counter += 1
                if self.max_runs is not None:
                    if counter > self.max_runs:
                        break

        # print('Learning:', self.agent_learning.keys())
        # for key, values in self.agent_learning.items():
        #     print(key, len(values))
        # print('Evaluation:', self.agent_evaluation.keys())
        # for key, values in self.agent_evaluation.items():
        #     print(key, len(values))

        labels = [
            # mpatches.Patch(color='black', label='Wait'),
            mpatches.Patch(color='red', label='Car'),
            mpatches.Patch(color='green', label='PT'),
            mpatches.Patch(color='orange', label='Walk'),
            mpatches.Patch(color='blue', label='Bicycle'),
            mpatches.Patch(color='gray', label='PTW'),
        ]

        modes_color = {
            0: 'black',
            1: 'red',
            2: 'green',
            3: 'orange',
            4: 'blue',
            5: 'gray',
        }

        fig, axs = plt.subplots(2, 3, figsize=(25, 15), constrained_layout=True)
        fig.suptitle('Evolution of agent "{}"'.format(self.agent_name))

        ############ LEARNIN
        axs[0][0].bar(self.agent['start']/3600, counter, width=0.05, color='g', align='center')
        axs[0][0].bar(9.0, counter, width=0.05, color='r', align='center')
        axs[0][0].set_xlim(6.5, 10.0)
        axs[0][0].set_ylim(-1, counter)
        axs[0][0].set_title('Learning')
        axs[0][0].set_xlabel('Time [h]')
        axs[0][0].set_ylabel('Training run [#]')
        # axs[0][0].legend(handles=labels)

        axs[1][0].bar(self.agent['start']/3600, counter, width=0.05, color='g', align='center')
        axs[1][0].bar(9.0, counter, width=0.05, color='r', align='center')
        axs[1][0].set_xlim(6.5, 10.0)
        axs[1][0].set_ylim(-1, counter)
        axs[1][0].set_title('Learning')
        axs[1][0].set_xlabel('Time [h]')
        axs[1][0].set_ylabel('Training run [#]')
        # axs[1][0].legend(handles=labels)

        for training_run, values in self.agent_learning.items():
            y = [training_run, training_run]
            rnd1 = random.randrange(len(values))
            rnd2 = random.randrange(len(values))
            while rnd1 == rnd2:
                rnd2 = random.randrange(len(values))
            ### RND 1
            episode = values[rnd1]
            travel = [episode['departure'], episode['arrival']]
            axs[0][0].plot(travel, y, color=modes_color[episode['mode']], alpha=0.75)
            ### RND 2
            episode = values[rnd2]
            travel = [episode['departure'], episode['arrival']]
            axs[1][0].plot(travel, y, color=modes_color[episode['mode']], alpha=0.75)

        ############ EVALUATION
        axs[0][1].bar(self.agent['start']/3600, counter, width=0.05, color='g', align='center')
        axs[0][1].bar(9.0, counter, width=0.05, color='r', align='center')
        axs[0][1].set_xlim(6.5, 10.0)
        axs[0][1].set_ylim(-1, counter)
        axs[0][1].set_title('Evaluation')
        axs[0][1].set_xlabel('Time [h]')
        # axs[0][1].legend(handles=labels)

        axs[1][1].bar(self.agent['start']/3600, counter, width=0.05, color='g', align='center')
        axs[1][1].bar(9.0, counter, width=0.05, color='r', align='center')
        axs[1][1].set_xlim(6.5, 10.0)
        axs[1][1].set_ylim(-1, counter)
        axs[1][1].set_title('Evaluation')
        axs[1][1].set_xlabel('Time [h]')
        # axs[1][1].legend(handles=labels)

        for training_run, values in self.agent_evaluation.items():
            y = [training_run, training_run]
            rnd1 = random.randrange(len(values))
            rnd2 = random.randrange(len(values))
            while rnd1 == rnd2:
                rnd2 = random.randrange(len(values))
            ### RND 1
            episode = values[rnd1]
            travel = [episode['departure'], episode['arrival']]
            axs[0][1].plot(travel, y, color=modes_color[episode['mode']], alpha=0.75)
            ### RND 2
            episode = values[rnd2]
            travel = [episode['departure'], episode['arrival']]
            axs[1][1].plot(travel, y, color=modes_color[episode['mode']], alpha=0.75)

        ############ OD
        axs[0][2].imshow(self.image, extent=[100, 3750, 0, 3600])
        axs[0][2].plot(*self.shape.exterior.xy)
        axs[0][2].plot(self.agent['origin'][0], self.agent['origin'][1], 'ko', color='g', alpha=0.99)
        axs[0][2].plot(self.agent['destination'][0], self.agent['destination'][1], 'ko', color='r', alpha=0.99)
        axs[0][2].plot(*self.shape.exterior.xy)
        axs[0][2].set_aspect('equal', 'box')
        axs[0][2].grid()

        ############# ??????????
        # imagebox = OffsetImage(self.image, zoom=0.25)
        # ab = AnnotationBbox(imagebox, (0.4, 0.5))
        # axs[1][2].set_xlim(0, 1)
        # axs[1][2].set_ylim(0, 1)
        # axs[1][2].add_artist(ab)
        axs[1][2].legend(handles=labels)

        #################################

        fig.savefig('{}.{}.svg'.format(self.prefix, self.agent_name),
                    dpi=300, transparent=False, bbox_inches='tight')
        # fig.savefig('{}.{}.png'.format(self.prefix, self.agent_name),
        #             dpi=300, transparent=False, bbox_inches='tight')

        # plt.show()
        matplotlib.pyplot.close('all')
        # sys.exit()

def _main():
    """ Process the RLLIB logs/result.json """

    config = _argument_parser()

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler = cProfile.Profile()
        profiler.enable()
    ## ========================              PROFILER              ======================== ##

    statistics = AgentStats(config.init, config.results, config.prefix,
                            config.agent, config.shapes, config.all_agents,
                            config.max, config.image)
    if config.all_agents:
        statistics.historical_info_all_agents()
    else:
        statistics.historical_info_single_agent()
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
