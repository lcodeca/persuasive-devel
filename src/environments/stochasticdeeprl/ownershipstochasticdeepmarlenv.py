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

from environments.stochasticdeeprl.complexstochasticdeepmarlenv import \
    ComplexStochasticPersuasiveDeepMARLEnv
from environments.stochasticdeeprl.ownershipcoopstochasticdeepmarlenv import \
    OwnershipDeepSUMOAgents

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
    logger.debug('[env_creator] Environment creation: OwnershipCSPersuasiveDeepMARLEnv')
    return OwnershipCSPersuasiveDeepMARLEnv(config)

####################################################################################################

#@ray.remote(num_cpus=10, num_gpus=1)
class OwnershipCSPersuasiveDeepMARLEnv(ComplexStochasticPersuasiveDeepMARLEnv):
    """ Simplified REWARD (with TRAVEL_TIME), aggregated MODE USAGE, FUTURE DEMAND and PARKING. """

    ################################################################################################

    def _initialize_agents(self):
        self.agents = dict()
        self.waiting_agents = list()
        for agent, agent_config in self.agents_init_list.items():
            self.waiting_agents.append((agent_config.start, agent))
            self.agents[agent] = OwnershipDeepSUMOAgents(agent_config, self.sumo_net)
        self.waiting_agents.sort()

    ################################################################################################

    def compute_info_for_agent(self, agent):
        """ Gather and return a dictionary containing the info associated with the given agent. """
        info = {
            'arrival': self.simulation.get_arrival(agent),
            'cost': self.agents[agent].cost,
            'departure': self.simulation.get_depart(agent),
            'discretized-cost': self.discrete_time(self.agents[agent].cost),
            'ett': self.agents[agent].ett,
            'mode': self.agents[agent].chosen_mode,
            'rtt': self.simulation.get_duration(agent),
            'wait': self.agents[agent].arrival - self.simulation.get_arrival(agent),
            'init': {
                'start': self.agents_init_list[agent].start,
                'origin': self.agents_init_list[agent].origin,
                'exp-arrival': self.agents_init_list[agent].exp_arrival[0],
                'arrival-slots-min': self._config['agent_init']['arrival-slots-min'],
                'ownership': self.agents[agent].ownership,
            }
        }
        logger.debug('Info for agent %s: \n%s', agent, pformat(info))
        return info

    ################################################################################################

    def step(self, action_dict):
        # action translation from array to value
        # for agent, action in action_dict.items():
        #     print(agent, action)
        obs, rewards, dones, infos = super(
            OwnershipCSPersuasiveDeepMARLEnv, self).step(action_dict)
        # print(pformat(obs))
        return obs, rewards, dones, infos

    ################################################################################################

    @staticmethod
    def deep_state_flattener(state):
        """ Flattening of the dictionary """
        deep = [
            state['origin_x'],
            state['origin_y'],
            state['destination_x'],
            state['destination_y'],
            state['time-left']
        ]
        deep.extend(state['ett'])
        deep.extend(state['usage'])
        deep.append(state['future_demand'])
        ownership = []
        for mode in sorted(state['ownership'].keys()):
            if state['ownership'][mode]:
                ownership.append(1)
            else:
                ownership.append(0)
        deep.extend(ownership)
        #####################################
        return deepcopy(deep)

    def craft_final_state(self, agent):
        final_state = collections.OrderedDict()
        final_state['origin_x'] = self.agents[agent].origin_x
        final_state['origin_y'] = self.agents[agent].origin_y
        final_state['destination_x'] = self.agents[agent].destination_x
        final_state['destination_y'] = self.agents[agent].destination_y
        final_state['time-left'] = self.discrete_time(0)
        final_state['ett'] = np.array([-1 for _ in self.agents[agent].modes])
        final_state['usage'] = np.array([-1 for _ in self.agents[agent].modes])
        final_state['future_demand'] = 0
        final_state['ownership'] = self.agents[agent].ownership
        observation = {
            "action_mask": [1] * self.get_action_space_size(agent),
            "avail_actions": [0] * self.get_action_space_size(agent),
            "obs": np.array(self.deep_state_flattener(final_state), dtype=np.float64),
        }
        return observation

    def get_observation(self, agent):
        """ Returns the observation of a given agent. """

        ##### Original observation
        ett = []
        now = self.simulation.get_current_time()
        if DEBUGGER:
            logger.debug('Time: \n%s', pformat(now))
        origin = self.agents[agent].origin
        destination = self.agents[agent].destination
        for mode in sorted(self.agents[agent].modes):
            _mode, _ptype, _vtype = self.simulation.get_mode_parameters(mode)
            try:
                route = self.simulation.traci_handler.simulation.findIntermodalRoute(
                    origin, destination, modes=_mode, pType=_ptype, vType=_vtype, depart=now,
                    routingMode=1)
                if DEBUGGER:
                    logger.debug('SUMO Route: \n%s', pformat(route))
                if not self.simulation.is_valid_route(mode, route):
                    route = None
            except (traci.exceptions.TraCIException, libsumo.libsumo.TraCIException):
                route = None
            if DEBUGGER:
                logger.debug('Filtered route: \n%s', pformat(route))
            if route:
                cost = self.simulation.cost_from_route(route)
                ett.append(self.discrete_time(cost))
                if self.agents[agent].ext:
                    self._add_observed_ett(agent, mode, cost)
            else:
                ett.append(self.discrete_time(self._config['scenario_config']['misc']['max_time']))
                if self.agents[agent].ext:
                    self._add_observed_ett(
                        agent, mode, self._config['scenario_config']['misc']['max_time'])

        timeleft = self.agents[agent].arrival - now

        ret = collections.OrderedDict()
        ret['from'] = self.edge_to_enum(self.agents[agent].origin)
        ret['to'] = self.edge_to_enum(self.agents[agent].destination)
        ret['time-left'] = self.discrete_time(timeleft)
        ret['ett'] = np.array(ett)
        if DEBUGGER:
            logger.debug('Observation: \n%s', pformat(ret))

        ##### MODES usage
        usage = []
        ret['origin_x'] = self.agents[agent].origin_x
        ret['origin_y'] = self.agents[agent].origin_y
        ret['destination_x'] = self.agents[agent].destination_x
        ret['destination_y'] = self.agents[agent].destination_y
        for mode in sorted(self.agents[agent].modes):
            agents_choice = self.episode_snapshot.get_history(mode)
            usage.append(self.agents_to_usage_active(agents_choice))
        ret['usage'] = usage

        ##### Future demand
        ret['future_demand'] = len(self.agents) - len(self.dones)

        ##### Ownership
        ret['ownership'] = self.agents[agent].ownership

        # Flattening of the dictionary
        deep_ret = self.deep_state_flattener(ret)

        # Observation space with Action masking
        # See: https://github.com/ray-project/ray/blob/releases/1.0.0/rllib/examples/env/parametric_actions_cartpole.py
        action_mask = [1] * self.get_action_space_size(agent)
        for mode, own in self.agents[agent].ownership.items():
            # print(mode, self.agents[agent].mode_to_action[mode], own)
            if not own:
                action_mask[self.agents[agent].mode_to_action[mode]] = 0
        # print(action_mask)
        observation = {
            "action_mask": action_mask,
            "avail_actions": [1] * self.get_action_space_size(agent),
            "obs": np.array(deep_ret, dtype=np.float64),
        }

        logger.debug('[%s] Observation: %s', agent, str(deep_ret))
        return observation

    def get_obs_space(self, agent):
        """ Returns the observation space. """
        parameters = 0
        parameters += 1                                 # from x coord
        parameters += 1                                 # from y coord
        parameters += 1                                 # to x coord
        parameters += 1                                 # to y coord
        parameters += 1                                 # time-left
        parameters += len(self.agents[agent].modes)     # ett
        parameters += len(self.agents[agent].modes)     # usage
        parameters += 1                                 # future demand
        parameters += len(self.agents[agent].ownership) # ownership

        lows = [
            self.bounding_box['bottom_left_X'],  # from x coord
            self.bounding_box['bottom_left_Y'],  # from y coord
            self.bounding_box['bottom_left_X'],  # to x coord
            self.bounding_box['bottom_left_Y'],  # to y coord
            0,  # time-left
        ]
        lows.extend([-1] * len(self.agents[agent].modes))    # ett
        lows.extend([-1] * len(self.agents[agent].modes))    # usage
        lows.append(0)                                       # future demand
        lows.extend([0] * len(self.agents[agent].ownership)) # ownership

        highs = [
            self.bounding_box['top_right_X'],  # from x coord
            self.bounding_box['top_right_Y'],  # from y coord
            self.bounding_box['top_right_X'],  # to x coord
            self.bounding_box['top_right_Y'],  # to y coord
            self.discrete_time(
                self._config['scenario_config']['misc']['max_time']),   # time-left in min
        ]
        highs.extend(
            [self.discrete_time(self._config['scenario_config']['misc']['max_time'])] *
            len(self.agents[agent].modes))                                      # ett
        highs.extend([len(self.agents)] * len(self.agents[agent].modes))        # usage
        highs.append(len(self.agents))                                          # future demand
        highs.extend([1] * len(self.agents[agent].ownership))                   # ownership

        logger.info('State space: [%d] - %s - %s', parameters, str(lows), str(highs))

        observation_space = gym.spaces.Dict({
            "action_mask": gym.spaces.Box(
                0, 1, shape=(self.get_action_space_size(agent), )),
            "avail_actions": gym.spaces.Box(
                0, 1, shape=(self.get_action_space_size(agent), )),
            "obs": gym.spaces.Box(
                low=np.array(lows), high=np.array(highs), dtype=np.float64)
        })
        return observation_space

    ################################################################################################

####################################################################################################
