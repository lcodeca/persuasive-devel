#!/usr/bin/env python3

""" Process a JSON worker file from RLLIB. """

import argparse
import collections
import cProfile
import io
import json
import logging
import os
from pprint import pformat, pprint
import pstats
from stat import S_ISREG, ST_CTIME, ST_MODE

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

def _argument_parser():
    """ Argument parser for the stats parser. """
    parser = argparse.ArgumentParser(
        description='RLLIB & SUMO Statistics parser.')
    parser.add_argument(
        '--input', required=True, type=str, help='Input JSON file or directory.')
    parser.add_argument(
        '--prefix', default='stats', help='Output prefix for the processed data.')
    parser.add_argument(
        '--max', default=None, type=int, help='Maximum number of agents to process.')
    parser.add_argument(
        '--profiler', dest='profiler', action='store_true', help='Enable cProfile.')
    parser.set_defaults(profiler=False)
    return parser.parse_args()

####################################################################################################

class JSONExplorer(object):
    """ Process a JSON worker file from RLLIB. """

    def __init__(self, filename, prefix, max_agents):
        self.input = filename
        self.prefix = prefix
        self.max = max_agents
        self.results = []
        self.agents = {}

        if os.path.isfile(self.input):
            self.process_json_file(self.input)
        else:
            self.walk_directory()
        self.plot_info_by_agent()

    def walk_directory(self):
        # get all entries in the directory
        entries = (os.path.join(self.input, file_name) for file_name in os.listdir(self.input))
        # Get their stats
        entries = ((os.stat(path), path) for path in entries)
        # leave only regular files, insert creation date
        entries = ((stat[ST_CTIME], path) for stat, path in entries if S_ISREG(stat[ST_MODE]))
        for _, entry in sorted(entries):
            self.process_json_file(entry)

    def process_json_file(self, filename):
        logger.info('Processing %s.', filename)
        with open(filename, 'r') as jsonfile:
            for row in tqdm(jsonfile): # enumerate cannot be used due to the size of the file
                complete = json.loads(row)
                agents = collections.defaultdict(dict)
                agent_id = complete['policy_batches']['unique']['agent_id']
                rewards = complete['policy_batches']['unique']['rewards']
                actions = complete['policy_batches']['unique']['actions']
                infos = complete['policy_batches']['unique']['infos']
                dones = complete['policy_batches']['unique']['dones']
                for pos, done in enumerate(dones):
                    if done:
                        agents[agent_id[pos]]['reward'] = rewards[pos]
                        agents[agent_id[pos]]['infos'] = infos[pos]
                        agents[agent_id[pos]]['actions'] = agent_id.count(agent_id[pos])
                        agents[agent_id[pos]]['mode'] = actions[pos]
                self.results.append(agents)

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

    def plot_info_by_agent(self):
        logger.info('Computing the detailed info for each agent over the episodes.')
        for episode in tqdm(self.results):
            for agent, report in episode.items():
                self._try_add_agent(agent)
                if agent not in self.agents:
                    continue

                self.agents[agent]['episode'].append(len(self.agents[agent]['episode']) + 1)
                self.agents[agent]['reward'].append(report['reward'])
                self.agents[agent]['actions'].append(report['actions'])
                self.agents[agent]['mode'].append(report['mode'])
                #############
                self.agents[agent]['cost'].append(report['infos']['cost']/60.0)
                self.agents[agent]['discretized-cost'].append(report['infos']['discretized-cost'])
                self.agents[agent]['rtt'].append(report['infos']['rtt']/60.0)
                #############
                self.agents[agent]['arrival'].append(report['infos']['arrival']/3600.0)
                self.agents[agent]['departure'].append(report['infos']['departure']/3600.0)
                self.agents[agent]['ett'].append(report['infos']['ett']/60.0)
                self.agents[agent]['wait'].append(report['infos']['wait']/60.0)
                #############
                self.agents[agent]['difference'].append(
                    (report['infos']['ett'] - report['infos']['rtt'])/60.0)

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

    def recursive_pprint(self, string, dictionary):
        if type(dictionary) is dict:
            input(dictionary.keys())
            for key, value in dictionary.items():
                self.recursive_pprint(key, value)
        else:
            print(string)
            input()
            pprint(dictionary)
            input()

def _main():
    """ Process a JSON worker file from RLLIB. """

    config = _argument_parser()

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler = cProfile.Profile()
        profiler.enable()
    ## ========================              PROFILER              ======================== ##

    statistics = JSONExplorer(config.input, config.prefix, config.max)
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
