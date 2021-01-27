#!/usr/bin/env python3

""" Stochastic MARL Environment based on PersuasiveDeepMARLEnv """

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

from environments.deeprl.deepmarlenvironment import (
    AgentsHistory, DeepSUMOAgents, DeepSUMOWrapper, PersuasiveDeepMARLEnv)

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
    logger.debug('[env_creator] Environment creation: StochasticPersuasiveDeepMARLEnv')
    return StochasticPersuasiveDeepMARLEnv(config)

####################################################################################################

# class StochasticDeepSUMOWrapper(DeepSUMOWrapper):

####################################################################################################

# class StochasticDeepSUMOAgents(DeepSUMOAgents):

####################################################################################################

class StochasticAgentsHistory(AgentsHistory):
    """ Stores the decisions made by the agents in a dictionary-like structure. """
    def __init__(self):
        self._data = defaultdict(list)

    def add_decision(self, action, time, agent_id):
        """ Save every decision made by the agents. """
        self._data[action].insert(0, (time, agent_id))

    def get_history(self, action):
        """ Return the complete history of all the decisions made by the agents. """
        return deepcopy(self._data[action])

    def __str__(self):
        """ Pretty-print the History """
        pretty = ''
        for action, history in self._data.items():
            pretty += '{} {}\n'.format(action, history)
        return pretty

#@ray.remote(num_cpus=10, num_gpus=1)
class StochasticPersuasiveDeepMARLEnv(PersuasiveDeepMARLEnv):
    """ Simplified REWARD, aggregated MODE USAGE, FUTURE DEMAND. """

    def __init__(self, config):
        """ Initialize the environment. """
        self.taz_shape = None
        self.episode_snapshot = StochasticAgentsHistory()
        self.bounding_box = {
            'bottom_left_X': config['scenario_config']['misc']['bounding_box'][0],
            'bottom_left_Y': config['scenario_config']['misc']['bounding_box'][1],
            'top_right_X': config['scenario_config']['misc']['bounding_box'][2],
            'top_right_Y': config['scenario_config']['misc']['bounding_box'][3],
        }
        super().__init__(config)
        self.tested_agents = False
        self.not_rewarded = set(self.agents.keys())

    def _generate_random_coords_in_area(self):
        """ Generates the random coords using the bounding box and fit them in the shape. """
        x_coord = self.rndgen.uniform(
            self.bounding_box['bottom_left_X'], self.bounding_box['top_right_X'])
        y_coord = self.rndgen.uniform(
            self.bounding_box['bottom_left_Y'], self.bounding_box['top_right_Y'])
        while not self.taz_shape.contains(geometry.Point(x_coord, y_coord)):
            x_coord = self.rndgen.uniform(
                self.bounding_box['bottom_left_X'], self.bounding_box['top_right_X'])
            y_coord = self.rndgen.uniform(
                self.bounding_box['bottom_left_Y'], self.bounding_box['top_right_Y'])
        return [x_coord, y_coord]

    def _config_from_agent_init(self):
        """ Load the agents configuration and format it in a usabe format. """
        # Convex hull initialization
        if not self.taz_shape:
            self.taz_shape = geometry.MultiPoint(
                self._config['agent_init']['taz-shape']).convex_hull

        # Extended Stats parameter
        ext_stats = self._config['agent_init']['ext-stats']

        # Starting time distribution:
        #   uniform during evaluation
        #   pareto during learning
        distribution = None
        if self._config['agent_init']['eval']:
            distribution = self.rndgen.uniform(
                self._config['agent_init']['eval']['from'],
                self._config['agent_init']['eval']['to'],
                self._config['agent_init']['agents']
            )
        else:
            distribution = self.rndgen.pareto(
                self._config['agent_init']['pareto-time-to-event']['a_shape'],
                self._config['agent_init']['agents'])
            distribution += 1
            distribution *= self._config['agent_init']['pareto-time-to-event']['m_min']
            # transform the time to the event to the starting time
            for pos, time_to_event in enumerate(distribution):
                _start = self._config['agent_init']['expected-arrival-time'][0]
                _start -= time_to_event * 60 # min to sec
                if _start < self._config['agent_init']['min_start']:
                    _start = self._config['agent_init']['min_start']
                distribution[pos] = _start

        if 'action-to-mode' not in self._config['agent_init']:
            self._config['agent_init']['action-to-mode'] = deepcopy(
                DeepSUMOAgents.default_action_to_mode)
        else:
            _tmp = {}
            for item, value in self._config['agent_init']['action-to-mode'].items():
                _tmp[int(item)] = value # JSON does not allows integers as keys, only strings
                self._config['agent_init']['action-to-mode'] = _tmp
        if 'modes-w-vehicles' not in self._config['agent_init']:
            self._config['agent_init']['modes-w-vehicles'] = deepcopy(
                DeepSUMOAgents.default_modes_w_vehicles)

        # Agents creation
        self.agents_init_list = dict()
        for agent_num in range(self._config['agent_init']['agents']):
            agent = 'agent_{}'.format(agent_num)
            current_ext_stats = False
            if ext_stats > 0:
                current_ext_stats = True
                ext_stats -= 1
            start = distribution[agent_num]
            origin = self._generate_random_coords_in_area()
            self.agents_init_list[agent] = DeepSUMOAgents.Config(
                agent, self._config['agent_init']['seed'],
                current_ext_stats, start, origin,
                self._config['agent_init']['destination'],
                self._config['agent_init']['modes'],
                self._config['agent_init']['action-to-mode'],
                self._config['agent_init']['modes-w-vehicles'],
                self._config['agent_init']['expected-arrival-time'])
            if DEBUGGER:
                logger.debug('%s', pformat(self.agents_init_list[agent]))

    def _initialize_agents(self):
        self.agents = dict()
        self.waiting_agents = list()
        for agent, agent_config in self.agents_init_list.items():
            self.waiting_agents.append((agent_config.start, agent))
            self.agents[agent] = DeepSUMOAgents(agent_config, self.sumo_net)
        self.waiting_agents.sort()

    def agents_to_usage_active(self, choices):
        """ Monitor active agents."""
        active = 0
        for _, agent in choices:
            if agent in self.simulation.subs:
                active += 1
        logger.debug('Usage: %d', active)
        return active

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
        return self.deep_state_flattener(final_state)

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

        # Flattening of the dictionary
        deep_ret = self.deep_state_flattener(ret)
        logger.debug('[%s] Observation: %s', agent, str(deep_ret))
        return np.array(deep_ret, dtype=np.int64)

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

        lows = [
            self.bounding_box['bottom_left_X'],  # from x coord
            self.bounding_box['bottom_left_Y'],  # from y coord
            self.bounding_box['bottom_left_X'],  # to x coord
            self.bounding_box['bottom_left_Y'],  # to y coord
            0,  # time-left
        ]
        lows.extend([-1] * len(self.agents[agent].modes))    # ett
        lows.extend([-1] * len(self.agents[agent].modes))    # usage
        lows.append(0) # future demand

        highs = [
            self.bounding_box['top_right_X'],  # from x coord
            self.bounding_box['top_right_Y'],  # from y coord
            self.bounding_box['top_right_X'],  # to x coord
            self.bounding_box['top_right_Y'],  # to y coord
            self.discrete_time(
                self._config['scenario_config']['misc']['max_time']),   # time-left in min
        ]
        highs.extend(                                                           # ett
            [self.discrete_time(self._config['scenario_config']['misc']['max_time'])] *
            len(self.agents[agent].modes))
        highs.extend([len(self.agents)] * len(self.agents[agent].modes)) # usage
        highs.append(len(self.agents)) # future demand

        logger.info('State space: [%d] - %s - %s', parameters, str(lows), str(highs))
        return gym.spaces.Box(
            low=np.array(lows), high=np.array(highs), dtype=np.float64)

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
            }
        }
        logger.debug('Info for agent %s: \n%s', agent, pformat(info))
        return info

    ################################################################################################

    def get_reward(self, agent):
        """ Return the reward for a given agent. """
        arrival = self.simulation.get_arrival(agent, default=None)
        max_penalty = self._config['scenario_config']['misc']['max_time'] / 60 # in minutes
        max_penalty /= self._config['agent_init']['arrival-slots-min'] # slotted time
        max_penalty *= self.agents[agent].late_weight

        #### ERRORS
        if arrival is None:
            logger.debug('Reward: Error for agent %s.', agent)
            return 0 - max_penalty

        #### TOO LATE
        if self.agents[agent].arrival < arrival:
            logger.debug('Reward: Agent %s arrived too late.', agent)
            penalty = arrival - self.agents[agent].arrival
            penalty /= 60 # in minutes
            penalty /= self._config['agent_init']['arrival-slots-min'] # slotted time
            penalty += 1 # late is always bad
            return 0 - penalty * self.agents[agent].late_weight

        arrival_buffer = (
            self.agents[agent].arrival - (self._config['agent_init']['arrival-slots-min'] * 60))
        #### TOO EARLY
        if arrival_buffer > arrival:
            logger.debug('Reward: Agent %s arrived too early.', agent)
            penalty = self.agents[agent].arrival - arrival
            penalty /= 60 # in minutes
            penalty /= self._config['agent_init']['arrival-slots-min'] # slotted time
            return 0 - penalty * self.agents[agent].waiting_weight

        #### ON TIME
        logger.info('Reward: Agent %s arrived on time.', agent)
        return 1

    ################################################################################################

    def reset(self):
        """ Resets the env and returns observations from ready agents. """
        logger.info('Multi-Agents Environment Reset')
        self.resetted = True
        self.environment_steps = 0
        self.episodes += 1
        self.active = set()
        self.dones = set()
        self.ext_stats = dict()

        # Reset the SUMO simulation
        if self.simulation:
            del self.simulation
        self.simulation = self.sumo_reset()

        # Reset the agents
        self._config_from_agent_init()
        self._initialize_agents()

        # Move the simulation forward
        starting_time = self.waiting_agents[0][0]
        self.simulation.fast_forward(starting_time)
        self._move_agents(starting_time)

        self.episode_snapshot = StochasticAgentsHistory()
        self.not_rewarded = set(self.agents.keys())
        feasible_plans = 0
        for agent in self.agents.values():
            if agent.test_agent(self.simulation):
                feasible_plans += 1
        logger.info('%d/%d agents have a feasible plan.', feasible_plans, len(self.agents))

        # Observations
        initial_obs = self.compute_observations(self.active)
        return initial_obs

    def step(self, action_dict):
        """
        Returns observations from ready agents.

        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.

        Returns
        -------
            obs (dict): New observations for each ready agent.
            rewards (dict): Reward values for each ready agent. If the
                episode is just started, the value will be None.
            dones (dict): Done values for each ready agent. The special key
                "__all__" (required) is used to indicate env termination.
            infos (dict): Optional info values for each agent id.
        """
        self.resetted = False
        self.environment_steps += 1
        logger.debug('========================> Episode: %d - Step: %d <==========================',
                     self.episodes, self.environment_steps)
        logger.debug('Actions: \n%s', pformat(action_dict))

        obs, rewards, dones, infos = {}, {}, {}, {}

        shuffled_agents = sorted(action_dict.keys()) # it may seem not smar to sort something that
                                                     # may need to be shuffled afterwards, but it
                                                     # is a matter of consistency instead of using
                                                     # whatever insertion order was used in the dict
        if self._config['scenario_config']['agent_rnd_order']:
            ## randomize the agent order to minimize SUMO's insertion queues impact
            logger.debug('Shuffling the order of the agents.')
            self.rndgen.shuffle(shuffled_agents) # in-place shuffle

        # Saves the episodes snapshots
        current_time = self.simulation.get_current_time()
        for agent, action in action_dict.items():
            if action == 0:
                # waiting
                continue
            self.episode_snapshot.add_decision(
                self.agents[agent].action_to_mode[action], current_time, agent)
        logger.debug('========================================================')
        logger.debug('Snapshot: \n%s', str(self.episode_snapshot))
        logger.debug('========================================================')

        # Take action
        wrong_decision = set()
        for agent in shuffled_agents:
            res = self.agents[agent].step(action_dict[agent], self.simulation)
            logger.debug('Agent %s returned %s', agent, pformat(res))
            if res == self.agents[agent].OUTCOME.INSERTED:
                logger.debug('Agent %s: INSERTED', agent)
                self.dones.add(agent)
                self.active.remove(agent)
            elif res == self.agents[agent].OUTCOME.WRONG_DECISION:
                logger.debug('Agent %s: WRONG_DECISION', agent)
                wrong_decision.add(agent)
                self.dones.add(agent)
                self.active.remove(agent)
            elif res == self.agents[agent].OUTCOME.WAITING:
                logger.debug('Agent %s: WAITING', agent)
            elif res == self.agents[agent].OUTCOME.ERROR:
                logger.debug('Agent %s: ERROR', agent)
                self.dones.add(agent)
                self.active.remove(agent)
                raise Exception(
                    'Agent {} has already been inserted in the simulation.'.format(agent))

        dones['__all__'] = len(self.dones) == len(self.agents)

        logger.debug('Before SUMO')
        ongoing_simulation = self.simulation.step(
            until_end=dones['__all__'], agents=dones['__all__'])
        logger.debug('After SUMO')

        # Compute rewards for the agents that are arrived to their destination.
        for agent in self.simulation.arrived_queue:
            logger.debug('Finalizing agent %s', agent)
            _state, _reward, _info = self.finalize_agent(agent)
            obs[agent] = _state
            rewards[agent] = _reward
            dones[agent] = True
            infos[agent] = _info
        self.simulation.arrived_queue = []

        # Punish all the agents that made a wrong decision
        for agent in wrong_decision:
            logger.debug('Punishing agent %s', agent)
            _state, _reward, _info = self.finalize_agent(agent)
            obs[agent] = _state
            rewards[agent] = _reward
            dones[agent] = True
            infos[agent] = _info

        waited_too_long = False
        if ongoing_simulation:
            current_time = self.simulation.get_current_time()

            ## add waiting agent to the pool of active agents
            logger.debug('Activating agents...')
            self._move_agents(current_time)

            # compute the new observation for the WAITING agents
            logger.debug('Computing obseravions for the WAITING agents.')
            for agent in self.active.copy():
                logger.debug('[%2f] %s --> %d', current_time, agent, self.agents[agent].arrival)
                if current_time > self.agents[agent].arrival:
                    waited_too_long = True
                    logger.warning('Agent %s waited for too long.', str(agent))
                    self.agents[agent].chosen_mode_error = (
                        'Waiting too long [{}]'.format(current_time))
                    self.agents[agent].ett = float('NaN')
                    self.agents[agent].cost = float('NaN')
                    self.dones.add(agent)
                    self.active.remove(agent)
                    _state, _reward, _info = self.finalize_agent(agent)
                    obs[agent] = _state
                    rewards[agent] = _reward
                    dones[agent] = True
                    infos[agent] = _info
                else:
                    rewards[agent] = 0
                    obs[agent] = self.get_observation(agent)
                    dones[agent] = False
                    infos[agent] = {}

        # in case all the reamining agents WAITED TOO LONG
        dones['__all__'] = len(self.dones) == len(self.agents)
        if waited_too_long and dones['__all__']:
            logger.info('All the agent are DONE. Finalizing episode...')
            ongoing_simulation = self.simulation.step(
                until_end=dones['__all__'], agents=dones['__all__'])
            for agent in self.simulation.arrived_queue:
                logger.debug('Finalizing agent %s', agent)
                _state, _reward, _info = self.finalize_agent(agent)
                obs[agent] = _state
                rewards[agent] = _reward
                dones[agent] = True
                infos[agent] = _info
            self.simulation.arrived_queue = []

        if not ongoing_simulation and not self.not_rewarded:
            logger.debug('Missing REWARD for [%d] %s',
                         len(self.not_rewarded), str(sorted(self.not_rewarded)))
            for agent in self.not_rewarded.copy():
                logger.debug('Finalizing agent %s', agent)
                _state, _reward, _info = self.finalize_agent(agent)
                obs[agent] = _state
                rewards[agent] = _reward
                dones[agent] = True
                infos[agent] = _info

        logger.debug('**********************************************************')
        logger.debug('==> self.dones: [%d] %s', len(self.dones), str(self.dones))
        logger.debug('==> self.active: [%d] %s', len(self.active), str(self.active))
        logger.debug('==> self.not_rewarded: [%d] %s', len(self.not_rewarded), str(self.not_rewarded))
        logger.debug('==> self.simulation.arrived_queue: [%d] %s', len(self.simulation.arrived_queue), str(self.simulation.arrived_queue))
        logger.debug('==> wrong_decision: [%d] %s', len(wrong_decision), str(wrong_decision))
        logger.debug('**********************************************************')

        logger.debug('Observations: %s', str(obs))
        logger.debug('Rewards: %s', str(rewards))
        logger.debug('Dones: %s', str(dones))
        logger.debug('Info: %s', str(infos))
        logger.debug('============================================================================')
        return obs, rewards, dones, infos

    ################################################################################################
