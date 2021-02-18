#!/usr/bin/env python3

"""
Stochastic MARL Environment based on ComplexStochasticPersuasiveDeepMARLEnv reward where the
passenger & ptw actions require finding a parking space.
"""

import collections
from enum import Enum
import logging
import os
import sys
import xml.etree.ElementTree

from collections import defaultdict
from copy import deepcopy
from pprint import pprint, pformat

import numpy as np
import shapely.geometry as geometry

import gym
import ray

from environments.stochasticdeeprl.complexparkingstochasticdeepmarlenv import \
    ComplexParkingDeepSUMOAgents
from environments.stochasticdeeprl.complexcoopstochasticdeepmarlenv import \
    CoopCSPersuasiveDeepMARLEnv

from utils.logger import set_logging

# """ Import SUMO library """
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
    import traci
    import libsumo
    import traci.constants as tc
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

####################################################################################################

DEBUGGER = True
logger = set_logging(__name__)

####################################################################################################

def env_creator(config):
    """ Environment creator used in the environment registration. """
    logger.debug('[env_creator] Environment creation: CoopCSParkingPersuasiveDeepMARLEnv')
    return CoopCSParkingPersuasiveDeepMARLEnv(config)

####################################################################################################

#@ray.remote(num_cpus=10, num_gpus=1)
class CoopCSParkingPersuasiveDeepMARLEnv(CoopCSPersuasiveDeepMARLEnv):
    """ Simplified REWARD (with TRAVEL_TIME), aggregated MODE USAGE, FUTURE DEMAND and PARKING. """

    def __init__(self, config):
        """ Initialize the environment. """
        # SUMO Parking location
        self._load_parking_from_additionals(config['scenario_config']['misc']['sumo_parking_file'])
        super().__init__(config)

    def _load_parking_from_additionals(self, filename):
        """ Load parkings ids from XML file. """
        self._sumo_parking = collections.defaultdict(list) # edge => list of ID
        if not os.path.isfile(filename):
            raise Exception("config['scenario_config']['misc']['sumo_parking_file'] is missing")
        xml_tree = xml.etree.ElementTree.parse(filename).getroot()
        for child in xml_tree:
            if child.tag != 'parkingArea':
                continue
            edge = child.attrib['lane'].split('_')[0]
            self._sumo_parking[edge].append(child.attrib['id'])

    def _initialize_agents(self):
        self.agents = dict()
        self.waiting_agents = list()
        for agent, agent_config in self.agents_init_list.items():
            self.waiting_agents.append((agent_config.start, agent))
            self.agents[agent] = ComplexParkingDeepSUMOAgents(
                agent_config, self.sumo_net, self._sumo_parking)
        self.waiting_agents.sort()

####################################################################################################
