#!/usr/bin/env python3

""" Generates JSON configuration file for a given scenario. """

import argparse
import collections
from copy import deepcopy
import itertools
import json
import os
import random
import sys

# """ Import SUMO library """
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
    import sumolib
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

####################################################################################################

def _argument_parser():
    """ Argument parser for the SUMOAgentGenerator. """
    parser = argparse.ArgumentParser(
        description='Generates JSON configuration file for a given scenario.')
    parser.add_argument('--net', required=True, type=str, help='The SUMO net.xml.')
    parser.add_argument('--num', default=10, type=int, help='Number of agent to be generated.')
    parser.add_argument('--out', default='output.agents.josm', type=str,
                        help='The JOSM output file.')
    parser.add_argument('--default', required=True, type=str,
                        help='The default agent settings (JOSM file).')
    parser.add_argument('--from-edge', default='', type=str, help='The default edge of origin.')
    parser.add_argument('--to-edge', default='', type=str, help='The default destination edge.')
    parser.add_argument('--all', dest='all', action='store_true',
                        help='Generate all the possible configurations.')
    parser.set_defaults(all=False)
    parser.add_argument('--random-seed', default=0, type=int, dest='seed',
                        help='If set, randomize a seed from 0 to the passed integer.')
    parser.add_argument('--random-start', nargs=2, type=int, default=[], dest='start',
                        help='If set, randomize the start using the passed interval.')
    return parser.parse_args()

####################################################################################################

class SUMOAgentGenerator(object):
    """ Generates JSON configuration file for a given scenario. """

    _config = None
    _sumo_net = None
    _edges = list()
    _agents = collections.defaultdict(dict)

    def __init__(self, config):
        """ config.net      --> 'The SUMO net.xml.'
            config.num      --> 'Number of agent to be generated.'
            config.out      --> 'The JOSM output file.'
            config.default  --> 'The default agent settings (JOSM).'
            config.from     --> 'The default edge of origin.'
            config.to       --> 'The default destination edge.'
            config.all      --> 'Generate all the possible configurations.'
            config.seed     --> 'If set, randomize a seed from 0 to the passed integer.
            config.start    --> 'If set, randomize the start using the passed interval.'
        """
        self._config = config
        self._load_edges()
        if config.all:
            self._generate_all_possible_agents()
        else:
            self._generate_agents()
        self._save_to_file()

    ################################################################################################

    def _load_edges(self):
        """ Load SUMO net.xml and extract the edges """
        self._sumo_net = sumolib.net.readNet(self._config.net)
        for edge in self._sumo_net.getEdges():
            if not edge.allows('pedestrian'):
                continue
            name = edge.getID()
            if ':' not in name:
                self._edges.append(name)

    def _generate_agents(self):
        """ Generate the agents """
        default = json.load(open(self._config.default))
        for agent in range(self._config.num):
            agent = 'agent_{}'.format(agent)
            if self._config.seed > 0:
                default['seed'] = random.randint(0, self._config.seed + 1)
            if self._config.start:
                begin, end = self._config.start
                default['start'] = random.randint(begin, end + 1)
            if self._config.from_edge:
                default['origin'] = self._config.from_edge
            else:
                default['origin'] = random.choice(self._edges)
            if self._config.to_edge:
                default['destination'] = self._config.to_edge
            else:
                default['destination'] = random.choice(self._edges)
            self._agents[agent] = deepcopy(default)

    def _generate_all_possible_agents(self):
        """ Generate all the possible agents """
        default = json.load(open(self._config.default))
        agent_counter = 0
        for from_edge, to_edge in itertools.permutations(self._edges, 2):
            if self._config.seed > 0:
                default['seed'] = random.randint(0, self._config.seed + 1)
            if self._config.start:
                begin, end = self._config.start
                default['start'] = random.randint(begin, end + 1)
            if self._config.from_edge and self._config.from_edge != from_edge:
                continue
            if self._config.to_edge and self._config.to_edge != to_edge:
                continue
            agent = 'agent_{}'.format(agent_counter)
            default['origin'] = from_edge
            default['destination'] = to_edge
            self._agents[agent] = deepcopy(default)
            agent_counter += 1

    ################################################################################################

    def _save_to_file(self):
        """ Save the agents in a JSON file. """
        json.dump(self._agents, open(self._config.out, 'w'))

    ################################################################################################

####################################################################################################

if __name__ == '__main__':

    cmd_args = _argument_parser()
    _ = SUMOAgentGenerator(cmd_args)
