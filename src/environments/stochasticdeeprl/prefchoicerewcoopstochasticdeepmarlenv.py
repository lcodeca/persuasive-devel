#!/usr/bin/env python3

"""
Stochastic MARL Environment based on CoopCSPersuasiveDeepMARLEnv reward where
the agents have preferences tied to the modes.
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
    logger.debug('[env_creator] Environment creation: PrefChoiceRefCoopCSPersuasiveDeepMARLEnv')
    return PrefChoiceRefCoopCSPersuasiveDeepMARLEnv(config)

####################################################################################################

#@ray.remote(num_cpus=10, num_gpus=1)
class PrefChoiceRefCoopCSPersuasiveDeepMARLEnv(CoopCSPersuasiveDeepMARLEnv):
    """ Simplified REWARD (with TRAVEL_TIME), aggregated MODE USAGE, FUTURE DEMAND. """

    def choice_modifier(self, agent, worst=False):
        """ Returns the correct choice modifier. """
        if worst:
            position = len(self.agents[agent].misc['rewards'])
            return self.agents[agent].misc['rewards'][str(position)]
        preferences = sorted([(weight, mode) for mode, weight in self.agents[agent].modes.items()])
        preferences = [mode for _, mode in preferences]
        position = preferences.index(self.agents[agent].chosen_mode)
        return self.agents[agent].misc['rewards'][str(position+1)]

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
            # Preference modifier
            pref_modifier = self.choice_modifier(agent, worst=True)
            return 0, 0 - (max_travel_time * 2) * self.agents[agent].late_weight * pref_modifier

        # real travel time
        travel_time = arrival - self.simulation.get_depart(agent) # travel time
        travel_time /= 60 # in minutes
        travel_time /= self._config['agent_init']['travel-slots-min'] # slotted time

        # Preference modifier
        pref_modifier = self.choice_modifier(agent)

        #### TOO LATE
        if self.agents[agent].arrival < arrival:
            logger.debug('Reward: Agent %s arrived too late.', agent)
            penalty = arrival - self.agents[agent].arrival # too late time
            penalty /= 60 # in minutes
            penalty /= self._config['agent_init']['travel-slots-min'] # slotted time
            if penalty <= 0:
                penalty += 1 # late is always bad
            return 0, 0 - (travel_time + penalty) * self.agents[agent].late_weight * pref_modifier

        arrival_buffer = (
            self.agents[agent].arrival - (self._config['agent_init']['arrival-slots-min'] * 60))

        #### TOO EARLY
        if arrival_buffer > arrival:
            logger.debug('Reward: Agent %s arrived too early.', agent)
            penalty = self.agents[agent].arrival - arrival
            penalty /= 60 # in minutes
            penalty /= self._config['agent_init']['travel-slots-min'] # slotted time
            return 1, 0 - (travel_time + penalty) * self.agents[agent].waiting_weight * pref_modifier

        #### ON TIME
        logger.info('Reward: Agent %s arrived on time.', agent)
        return 1, 1 - travel_time * pref_modifier

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
                'preferences': self.agents[agent].modes,
            }
        }
        logger.debug('Info for agent %s: \n%s', agent, pformat(info))
        return info

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
        deep.extend(state['preferences'])
        # print('Flattned observation: ', deep)
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

        ##### Preferences
        preferences = []
        for mode in sorted(self.agents[agent].modes):
            preferences.append(self.agents[agent].modes[mode])
        final_state['preferences'] = preferences
        # print('Preferences:', preferences)

        observation = np.array(
            self.deep_state_flattener(final_state), dtype=np.float64)
        logger.debug('[%s] Final observation: %s', agent, pformat(observation))
        # print('[{}] Final observation: {}'.format(agent, pformat(observation)))
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

        ##### Preferences
        preferences = []
        for mode in sorted(self.agents[agent].modes):
            preferences.append(self.agents[agent].modes[mode])
        ret['preferences'] = preferences
        # print('[{}] Preferences: {}'.format(agent, preferences))

        # Flattening of the dictionary
        deep_ret = self.deep_state_flattener(ret)

        observation = np.array(deep_ret, dtype=np.float64)

        # print('[{}] Observation: {}'.format(agent, pformat(observation)))
        logger.debug('[%s] Observation: %s', agent, pformat(observation))
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
        parameters += len(self.agents[agent].modes)     # preferences

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
        lows.extend([0.5] * len(self.agents[agent].modes)) # preferences

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
        highs.extend([1.5] * len(self.agents[agent].modes))                     # preferences

        logger.info('State space: [%d] - %s - %s', parameters, str(lows), str(highs))

        observation_space = gym.spaces.Box(
            low=np.array(lows),
            high=np.array(highs),
            dtype=np.float64)
        return observation_space

    ################################################################################################

####################################################################################################
