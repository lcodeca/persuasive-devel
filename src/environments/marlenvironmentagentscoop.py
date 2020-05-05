#!/usr/bin/env python3

""" In itial Agents Cooperation MARL Environment based on PersuasiveMultiAgentEnv """

import logging
import os
import sys

from collections import defaultdict
from copy import deepcopy
from pprint import pformat

import gym
import numpy as np
from numpy.random import RandomState

from ray.rllib.env import MultiAgentEnv
from rllibsumoutils.sumoutils import SUMOUtils

from environments.marlenvironment import PersuasiveMultiAgentEnv

# """ Import SUMO library """
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
    from traci.exceptions import TraCIException
    import traci.constants as tc
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

####################################################################################################

DEBUGGER = True
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

####################################################################################################

def env_creator(config):
    """ Environment creator used in the environment registration. """
    LOGGER.debug('[env_creator] Environment creation: AgentsCoopMultiAgentEnv')
    return AgentsCoopMultiAgentEnv(config)

####################################################################################################

class AgentsDecisions(object):
    """ Stores the decisions made by the agents in a dictionary-like structure. """
    def __init__(self):
        self._data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))
    
    def add_decision(self, origin, destination, action, time, agent_id):
        """ """
        self._data[origin][destination][action].insert(0, (time, agent_id))
        
    def get_history(self, origin, destination, action):
        """ """
        return deepcopy(self._data[origin][destination][action])
    
    def __str__(self):
        pretty = ''
        for origin, vals1 in self._data.items():
            for dest, vals2 in vals1.items():
                for mode, series in vals2.items():
                        pretty += '{} {} {} {}\n'.format(origin, dest, mode, series)
        return pretty

class AgentsCoopMultiAgentEnv(PersuasiveMultiAgentEnv):
    """ Initial implementation of Agents Cooperation based on the PersuasiveMultiAgentEnv. """

    def __init__(self, config):
        """ Initialize the environment. """
        super().__init__(config)
        self.episode_snapshot = AgentsDecisions()

    def agents_to_usage_active(self, choices):
        """ """
        # filter only the agents still in the simulation // this should be optimized
        people = set(self.simulation.traci_handler.person.getIDList())
        active = 0
        for _, agent in choices:
            if agent in people:
                active += 1
        ## number of agents that choose the mode and are still in the sim
        ## divided by the total number of agents
        ## multiplied by 100 to have a percentage
        ## divided by the level of usage we intend to cover
        ret = round(active / len(self.agents) * 100 / 10)
        LOGGER.debug('Usage: %d / %d * 100 / 10 = %d (rounded).', active, len(self.agents), ret)
        return ret

    def get_observation(self, agent):
        """ Returns the observation of a given agent. """
        ret = super().get_observation(agent)
        usage = []
        origin = self.agents[agent].origin
        destination = self.agents[agent].destination
        for mode in sorted(self.agents[agent].modes):
            agents_choice = self.episode_snapshot.get_history(origin, destination, mode)
            usage.append(self.agents_to_usage_active(agents_choice))
        ret['usage'] = usage
        if DEBUGGER:
            LOGGER.debug('Observation: \n%s', pformat(ret))
        return ret

    ################################################################################################

    def reset(self):
        """ Resets the env and returns observations from ready agents. """
        initial_obs = super().reset()
        self.episode_snapshot = AgentsDecisions()
        return initial_obs

    def step(self, action_dict):
        """
        Returns observations from ready agents.
        """
        now = self.simulation.get_current_time()
        for agent, action in action_dict.items():
            if action == 0:
                # waiting
                continue
            self.episode_snapshot.add_decision(
                self.agents[agent].origin,
                self.agents[agent].destination,
                self.agents[agent].action_to_mode[action],
                now, agent)
        LOGGER.debug('========================================================')
        LOGGER.debug('Snapshot: \n%s', str(self.episode_snapshot))
        LOGGER.debug('========================================================')
        return super().step(action_dict)

    def craft_final_state(self, agent):
        final_state = super().craft_final_state(agent)
        final_state['usage'] = np.array([-1 for _ in self.agents[agent].modes])
        return final_state

    ################################################################################################

    def get_obs_space_size(self, agent):
        """ Returns the size of the observation space. """
        return (len(self._edges_to_int) *                                                                # from
                len(self._edges_to_int) *                                                                # to
                self._config['scenario_config']['misc']['max-time'] *                                    # time to event
                (self._config['scenario_config']['misc']['max-time'] * len(self.agents[agent].modes)) *  # ETT by mode
                (10 * len(self.agents[agent].modes)))                                                    # Usage by mode

    def get_obs_space(self, agent):
        """ Returns the observation space. """
        return gym.spaces.Dict({
            'from': gym.spaces.Discrete(len(self._edges_to_int)),
            'to': gym.spaces.Discrete(len(self._edges_to_int)),
            'time-left': gym.spaces.Discrete(self._config['scenario_config']['misc']['max-time']),
            'ett': gym.spaces.MultiDiscrete(
                [self._config['scenario_config']['misc']['max-time']] * (len(self.agents[agent].modes))),
            'usage': gym.spaces.MultiDiscrete([10] * (len(self.agents[agent].modes))),
        })    

    ################################################################################################