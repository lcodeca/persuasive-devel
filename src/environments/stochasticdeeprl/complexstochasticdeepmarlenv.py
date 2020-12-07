#!/usr/bin/env python3

""" Stochastic MARL Environment based on StochasticPersuasiveDeepMARLEnv """

import collections
from enum import Enum
import logging
import os
import sys

from collections import defaultdict
from copy import deepcopy
from pprint import pprint, pformat

import numpy as np
import shapely.geometry as geometry

import gym
import ray

from environments.deeprl.deepmarlenvironment import DeepSUMOAgents
from environments.stochasticdeeprl.stochasticdeepmarlenv import StochasticPersuasiveDeepMARLEnv

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
    logger.debug('[env_creator] Environment creation: ComplexStochasticPersuasiveDeepMARLEnv')
    return ComplexStochasticPersuasiveDeepMARLEnv(config)

####################################################################################################

class ComplexDeepSUMOAgents(DeepSUMOAgents):
    """ SUMO agent that computes specific max penalties given origin and destination. """

    def compute_max_travel_time(self, handler):
        max_travel_time = None
        for mode in self.modes:
            _mode, _ptype, _vtype = handler.get_mode_parameters(mode)
            route = None
            try:
                route = handler.traci_handler.simulation.findIntermodalRoute(
                    self.origin, self.destination, modes=_mode, pType=_ptype, vType=_vtype,
                    routingMode=1)
                if not handler.is_valid_route(mode, route):
                    route = None
            except (traci.exceptions.TraCIException, libsumo.libsumo.TraCIException):
                route = None
            if route is not None:
                ett = 0.0
                for stage in list(route):
                    ett += stage.travelTime
                if max_travel_time is None:
                    max_travel_time = ett
                else:
                    max_travel_time = max(max_travel_time, ett)
        if max_travel_time is None:
            logger.error('Broken routes for max penalty computation for agent %s', self.agent_id)
            logger.error('Parameters: %s --> %s, [%s].', self.origin, self.destination, self.modes)
            raise Exception(pformat(self))
        self.max_travel_time = max_travel_time

####################################################################################################

#@ray.remote(num_cpus=10, num_gpus=1)
class ComplexStochasticPersuasiveDeepMARLEnv(StochasticPersuasiveDeepMARLEnv):
    """ Simplified REWARD (with TRAVEL_TIME), aggregated MODE USAGE, FUTURE DEMAND. """

    def _initialize_agents(self):
        self.agents = dict()
        self.waiting_agents = list()
        for agent, agent_config in self.agents_init_list.items():
            self.waiting_agents.append((agent_config.start, agent))
            self.agents[agent] = ComplexDeepSUMOAgents(agent_config, self.sumo_net)
        self.waiting_agents.sort()

    ############################################################################

    def reset(self):
        """ Resets the env and returns observations from ready agents. """
        initial_obs = super().reset()
        for agent in self.agents.values():
            agent.compute_max_travel_time(self.simulation)
        return initial_obs

    ############################################################################

    def get_reward(self, agent):
        """ Return the reward for a given agent. """

        arrival = self.simulation.get_arrival(agent, default=None)
        #### ERRORS
        if arrival is None:
            logger.warning('Reward: Error for agent %s.', agent)
            # the maximum penalty is set as the slowest travel time,
            #   while starting when is too late.
            max_travel_time = self.agents[agent].max_travel_time
            max_travel_time /= self._config['agent_init']['travel-slots-min'] # slotted time
            return 0 - (max_travel_time * 2) * self.agents[agent].late_weight

        # real travel time
        travel_time = arrival - self.simulation.get_depart(agent) # travel time
        travel_time /= 60 # in minutes
        travel_time /= self._config['agent_init']['travel-slots-min'] # slotted time

        #### TOO LATE
        if self.agents[agent].arrival < arrival:
            logger.debug('Reward: Agent %s arrived too late.', agent)
            penalty = arrival - self.agents[agent].arrival # too late time
            penalty /= 60 # in minutes
            penalty /= self._config['agent_init']['travel-slots-min'] # slotted time
            if penalty <= 0:
                penalty += 1 # late is always bad
            return 0 - (travel_time + penalty) * self.agents[agent].late_weight

        arrival_buffer = (
            self.agents[agent].arrival - (self._config['agent_init']['arrival-slots-min'] * 60))

        #### TOO EARLY
        if arrival_buffer > arrival:
            logger.debug('Reward: Agent %s arrived too early.', agent)
            penalty = self.agents[agent].arrival - arrival
            penalty /= 60 # in minutes
            penalty /= self._config['agent_init']['travel-slots-min'] # slotted time
            return 0 - (travel_time + penalty) * self.agents[agent].waiting_weight

        #### ON TIME
        logger.info('Reward: Agent %s arrived on time.', agent)
        return 1 - travel_time

    ################################################################################################
