#!/usr/bin/env python3

""" Initial Agents Cooperation MARL Environment based on PersuasiveMultiAgentEnv """

import logging
import os
import sys

from collections import defaultdict
from copy import deepcopy
from pprint import pprint, pformat

import numpy as np

import gym
import ray

from environments.rl.marlenvironment import (
    PersuasiveMultiAgentEnv, SUMOModeAgent, SUMOSimulationWrapper)
from rllibsumoutils.sumoutils import sumo_default_config, SUMOUtils

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
    logger.debug('[env_creator] Environment creation: PersuasiveDeepMARLEnv')
    return PersuasiveDeepMARLEnv(config)

####################################################################################################

class DeepSUMOWrapper(SUMOUtils):

    def _initialize_simulation(self):
        """ Specific simulation initialization. """
        try:
            super()._initialize_simulation()
        except NotImplementedError:
            pass

    def _initialize_metrics(self):
        """ Specific metrics initialization """
        try:
            super()._initialize_metrics()
        except NotImplementedError:
            pass
        self.agents_depart = dict()
        self.agents_arrival = dict()
        self.arrived_queue = list()

    def _default_step_action(self, agents):
        """ Specific code to be executed in every simulation step """
        _dones, _agents = None, None
        try:
            _dones, _agents = agents
        except Exception as exception:
            logger.error('%s', str(agents))
            raise exception
        dones, agents = _dones, _agents
        try:
            super()._default_step_action(agents)
        except NotImplementedError:
            pass
        now = self.traci_handler.simulation.getTime() # seconds
        logger.debug('[%.2f] Agents: %s', now, str(sorted(agents)))
        people = self.traci_handler.person.getIDList()
        logger.debug('[%.2f] People: %s', now, str(sorted(people)))
        left = set()
        for agent in agents:
            if agent in people:
                if agent not in self.agents_depart:
                    self.agents_depart[agent] = now
                    logger.debug('[%.2f] Agent %s has departed.', now, agent)
            else:
                if agent in self.agents_depart and agent not in self.agents_arrival:
                    self.agents_arrival[agent] = now
                    logger.debug('[%.2f] Agent %s has arrived.', now, agent)
                    self.arrived_queue.append(agent)
                if dones:
                    left.add(agent)
        logger.debug('[%.2f] Left: %d - %s', now, len(left), sorted(left))
        logger.debug('[%.2f] Agents: %d', now, len(agents))
        if dones:
            if len(agents) == len(left):
                # all the agents left the simulation
                logger.info('[%.2f] All the AGENTS left the SUMO simulation.', now)
                self.end_simulation()
        return True

    def end_simulation(self):
        """ Forces the simulation to stop. """
        if self.is_ongoing_sim():
            logger.info('[%.2f] Closing TraCI %s',
                        self.traci_handler.simulation.getTime(), self._sumo_label)
            self._manually_stopped = True
            self.traci_handler.close()
        else:
            logger.warning('TraCI %s is already closed.', self._sumo_label)

    def get_depart(self, entity, default=float('NaN')):
        """ Override the TRIPINFO function. """
        if entity in self.agents_depart:
            return self.agents_depart[entity]
        return default

    def get_arrival(self, entity, default=float('NaN')):
        """ Override the TRIPINFO function. """
        if entity in self.agents_arrival:
            return self.agents_arrival[entity]
        return default

    def get_duration(self, entity, default=float('NaN')):
        """ Override the TRIPINFO function. """
        if entity in self.agents_arrival and entity in self.agents_depart:
            return self.get_arrival(entity) - self.get_depart(entity)
        return default

    def get_penalty_time(self):
        """ Return the penalty max time. """
        return self._config['misc']['max_time']

    def process_tripinfo_file(self):
        """
            Closes the TraCI connections, then reads and process the tripinfo data.
            It requires 'tripinfo_xml_file' and 'tripinfo_xml_schema' configuration parametes set.
        """
        processed = False
        counter = 0
        while not processed and counter < 10:
            try:
                super().process_tripinfo_file()
                processed = True
            except Exception as exception:
                logger.error('[%d] %s', os.getpid(), str(exception))
                counter += 1

####################################################################################################

DEFAULT_AGENT_CONFING = {
    'origin': [485.72, 2781.14],
    'destination': [2190.42, 1797.96],
    'start': 0,
    'expected-arrival-time': [32400, 2.0, 4.0],
        'modes': {
            'passenger': 1.0,
            'public': 1.0,
            'walk': 1.0,
            'bicycle': 1.0,
            'ptw': 1.0,
            'on-demand': 1.0
        },
    'seed': 42,
    'init': [0, 0],
    'ext-stats': False
}

class DeepSUMOAgents(SUMOModeAgent):
    """ SUMO agent that translate x,y coords to edges to be used in the simulation. """

    def __init__(self, config, network):
        """ Initialize the environment. """
        super().__init__(config)
        self.origin_x, self.origin_y = self.origin
        edges = network.getNeighboringEdges(
            self.origin_x, self.origin_y, r=1000,
            includeJunctions=False, allowFallback=True)
        self.origin = None
        for distance, edge in sorted([(dist, edge) for edge, dist in edges]):
            if edge.allows('pedestrian'):
                self.origin = edge.getID()
                if distance > 500:
                    logger.warning(
                        '[%s] Origin %.2f, %.2f is %.2f from edge %s',
                        self.agent_id, self.origin_x, self.origin_y, distance, self.origin)
                break
        if self.origin is None:
            raise Exception('Origin not foud for agent {}'.format(self.agent_id))

        self.destination_x, self.destination_y = self.destination
        edges = network.getNeighboringEdges(
            self.destination_x, self.destination_y, r=1000,
            includeJunctions=False, allowFallback=True)
        self.destination = None
        for distance, edge in sorted([(dist, edge) for edge, dist in edges]):
            if edge.allows('pedestrian'):
                self.destination = edge.getID()
                if distance > 500:
                    logger.warning(
                        '[%s] Destination %.2f, %.2f is %.2f from edge %s',
                        self.agent_id, self.destination_x, self.destination_y, distance,
                        self.destination)
                break
        if self.destination is None:
            raise Exception('Destination not foud for agent {}'.format(self.agent_id))

    def test_agent(self, handler):
        feasible_plan = False
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
                feasible_plan = True
        if not feasible_plan:
            logging.error('No feasible route for agent %s from %s to %s with modes %s.',
                          self.agent_id, self.origin, self.destination, str(self.modes))
        return feasible_plan

####################################################################################################

class AgentsHistory(object):
    """ Stores the decisions made by the agents in a dictionary-like structure. """
    def __init__(self):
        self._data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))

    def add_decision(self, origin, destination, action, time, agent_id):
        """ Save every decision made by the agents. """
        self._data[origin][destination][action].insert(0, (time, agent_id))

    def get_history(self, origin, destination, action):
        """ Return the complete history of all the decisions made by the agents. """
        return deepcopy(self._data[origin][destination][action])

    def __str__(self):
        """ Pretty-print the History """
        pretty = ''
        for origin, vals1 in self._data.items():
            for dest, vals2 in vals1.items():
                for mode, series in vals2.items():
                    pretty += '{} {} {} {}\n'.format(origin, dest, mode, series)
        return pretty

DEFAULT_SCENARIO_CONFING = {
    'sumo_config': sumo_default_config(),
    'agent_rnd_order': True,
    'log_level': 'WARN',
    'seed': 42,
    'misc': {
        'sumo_net_file': '',
        'bounding_box': [-146.63, -339.13, 4043.48, 3838.63],
        'max_time': 86400
    }
}

#@ray.remote(num_cpus=10, num_gpus=1)
class PersuasiveDeepMARLEnv(PersuasiveMultiAgentEnv):
    """
    Initial implementation of Late Reward and Agents Cooperation based on the
    PersuasiveMultiAgentEnv, explicitly for deep learning implementation. """

    def __init__(self, config):
        """ Initialize the environment. """
        super().__init__(config)
        self.episode_snapshot = AgentsHistory()
        self.bounding_box = {
            'bottom_left_X': config['scenario_config']['misc']['bounding_box'][0],
            'bottom_left_Y': config['scenario_config']['misc']['bounding_box'][1],
            'top_right_X': config['scenario_config']['misc']['bounding_box'][2],
            'top_right_Y': config['scenario_config']['misc']['bounding_box'][3],
        }
        self.tested_agents = False
        self.not_rewarded = set(self.agents.keys())

    def _initialize_agents(self):
        self.agents = dict()
        self.waiting_agents = list()
        for agent, agent_config in self.agents_init_list.items():
            self.waiting_agents.append((agent_config.start, agent))
            self.agents[agent] = DeepSUMOAgents(agent_config, self.sumo_net)
        self.waiting_agents.sort()

    def agents_to_usage_active(self, choices):
        """ Monitor active agents."""
        # filter only the agents still in the simulation // this should be optimized
        people = set(self.simulation.traci_handler.person.getIDList())
        active = 0
        for _, agent in choices:
            if agent in people:
                active += 1
        logger.debug('Usage: %d', active)
        return active

    ################################################################################################

    def sumo_reset(self):
        logger.critical('PersuasiveDeepMARLEnv.sumo_reset: PID %s', os.getpid())
        return DeepSUMOWrapper(self._config['scenario_config']['sumo_config'])

    def reset(self):
        """ Resets the env and returns observations from ready agents. """
        initial_obs = super().reset()
        self.episode_snapshot = AgentsHistory()
        self.not_rewarded = set(self.agents.keys())
        if not self.tested_agents:
            feasible_plans = 0
            for agent in self.agents.values():
                if agent.test_agent(self.simulation):
                    feasible_plans += 1
            logger.info('%d/%d agents have a feasible plan.', feasible_plans, len(self.agents))
            self.tested_agents = True
        return initial_obs

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
            'wait': self.agents[agent].arrival - self.simulation.get_arrival(agent)
        }
        logger.debug('Info for agent %s: \n%s', agent, pformat(info))
        return info

    def finalize_agent(self, agent):
        """ Gather all the required ifo and aggregate the rewards for the agents. """
        self.not_rewarded.remove(agent)
        state = self.craft_final_state(agent)
        reward = self.get_reward(agent)
        info = self.compute_info_for_agent(agent)
        logger.debug('Observations: %s', str(state))
        logger.debug('Rewards: %s', str(reward))
        logger.debug('Info: %s', str(info))
        return state, reward, info

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
                self.agents[agent].origin,
                self.agents[agent].destination,
                self.agents[agent].action_to_mode[action], current_time, agent)
        logger.debug('========================================================')
        logger.debug('Snapshot: \n%s', str(self.episode_snapshot))
        logger.debug('========================================================')

        # Take action
        for agent in shuffled_agents:
            done = self.agents[agent].step(action_dict[agent], self.simulation)
            if done:
                self.dones.add(agent)
                self.active.remove(agent)
        dones['__all__'] = len(self.dones) == len(self.agents)

        logger.debug('Before SUMO')
        ongoing_simulation = self.simulation.step(
            until_end=dones['__all__'],
            agents=(dones['__all__'], set(self.agents.keys())))
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

        waited_too_long = False
        if ongoing_simulation:
            ## add waiting agent to the pool of active agents
            current_time = self.simulation.get_current_time()
            logger.debug('Activating agents...')
            self._move_agents(current_time)

            # compute the new observation for the WAITING agents
            logger.debug('Computing obseravions for the WAITING agents.')
            agents_to_remove = set()
            for agent in self.active:
                logger.debug('[%2f] %s --> %d', current_time, agent, self.agents[agent].arrival)
                if current_time > self.agents[agent].arrival:
                    waited_too_long = True
                    logger.warning('Agent %s waited for too long.', str(agent))
                    self.agents[agent].chosen_mode_error = (
                        'Waiting too long [{}]'.format(current_time))
                    self.agents[agent].ett = float('NaN')
                    self.agents[agent].cost = float('NaN')
                    self.dones.add(agent)
                    agents_to_remove.add(agent)
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
            for agent in agents_to_remove:
                self.active.remove(agent)

        # in case all the reamining agents WAITED TOO LONG
        dones['__all__'] = len(self.dones) == len(self.agents)
        if waited_too_long and dones['__all__']:
            logger.info('All the agent are DONE. Finalizing episode...')
            ongoing_simulation = self.simulation.step(
                until_end=dones['__all__'],
                agents=(dones['__all__'], set(self.agents.keys())))
            for agent in self.simulation.arrived_queue:
                logger.debug('Finalizing agent %s', agent)
                _state, _reward, _info = self.finalize_agent(agent)
                obs[agent] = _state
                rewards[agent] = _reward
                dones[agent] = True
                infos[agent] = _info
            self.simulation.arrived_queue = []

        if not ongoing_simulation:
            logger.debug('Missing REWARD for [%d] %s',
                         len(self.not_rewarded), str(sorted(self.not_rewarded)))
            for agent in self.not_rewarded.copy():
                logger.debug('Finalizing agent %s', agent)
                _state, _reward, _info = self.finalize_agent(agent)
                obs[agent] = _state
                rewards[agent] = _reward
                dones[agent] = True
                infos[agent] = _info
            logger.debug('Missing REWARD for [%d] %s',
                         len(self.not_rewarded), str(sorted(self.not_rewarded)))

        logger.debug('Observations: %s', str(obs))
        logger.debug('Rewards: %s', str(rewards))
        logger.debug('Dones: %s', str(dones))
        logger.debug('Info: %s', str(infos))
        logger.debug('============================================================================')
        return obs, rewards, dones, infos

    ################################################################################################

    def get_reward(self, agent):
        """ Return the reward for a given agent. """
        if not self.agents[agent].chosen_mode:
            logger.warning('Agent %s mode error: "%s"', agent, self.agents[agent].chosen_mode_error)
            return 0 - int(self.simulation.get_penalty_time())

        journey_time = self.simulation.get_duration(agent)
        if np.isnan(journey_time):
            ## This should never happen.
            ## If it does, there is a bug/issue with the SUMO/MARL environment interaction.
            logger.error('No journey time for %s.', agent)
            return 0 - int(self.simulation.get_penalty_time())
        logger.debug(' Agent: %s, journey: %s', agent, str(journey_time))
        arrival = self.simulation.get_arrival(agent,
                                              default=self.simulation.get_penalty_time())
        logger.debug(' Agent: %s, arrival: %s', agent, str(arrival))

        # REWARD = journey time * mode weight + ....
        reward = journey_time * self.agents[agent].modes[self.agents[agent].chosen_mode]
        if self.agents[agent].arrival < arrival:
            ## agent arrived too late
            late_time = arrival - self.agents[agent].arrival
            reward += late_time * self.agents[agent].late_weight
            logger.debug('Agent: %s, arrival: %s, wanted arrival: %s, late: %s',
                         agent, str(arrival), str(self.agents[agent].arrival), str(late_time))
        elif self.agents[agent].arrival > arrival:
            ## agent arrived too early
            waiting_time = self.agents[agent].arrival - arrival
            reward += waiting_time * self.agents[agent].waiting_weight
            logger.debug('Agent: %s, duration: %s, waiting: %s, wanted arrival: %s',
                         agent, str(journey_time), str(waiting_time), str(arrival))
        else:
            logger.debug('Agent: %s it is perfectly on time!', agent)

        return int(0 - (reward))

    ################################################################################################

    @staticmethod
    def deep_state_flattener(state):
        # Flattening of the dictionary
        deep = [
            # state['from'],
            # state['to'],
            state['origin_x'],
            state['origin_y'],
            state['destination_x'],
            state['destination_y'],
            state['time-left']
        ]
        deep.extend(state['ett'])
        deep.extend(state['usage'])
        return deepcopy(deep)

    def craft_final_state(self, agent):
        final_state = super().craft_final_state(agent)
        final_state['origin_x'] = self.agents[agent].origin_x
        final_state['origin_y'] = self.agents[agent].origin_y
        final_state['destination_x'] = self.agents[agent].destination_x
        final_state['destination_y'] = self.agents[agent].destination_y
        final_state['usage'] = np.array([-1 for _ in self.agents[agent].modes])
        return self.deep_state_flattener(final_state)

    def discrete_time(self, time_s):
        if np.isnan(time_s):
            return time_s
        return int(round((time_s / 60.0), 0))

    def get_observation(self, agent):
        """ Returns the observation of a given agent. """
        ret = super().get_observation(agent)
        usage = []
        ret['origin_x'] = self.agents[agent].origin_x
        ret['origin_y'] = self.agents[agent].origin_y
        ret['destination_x'] = self.agents[agent].destination_x
        ret['destination_y'] = self.agents[agent].destination_y
        origin = self.agents[agent].origin
        destination = self.agents[agent].destination
        for mode in sorted(self.agents[agent].modes):
            agents_choice = self.episode_snapshot.get_history(origin, destination, mode)
            usage.append(self.agents_to_usage_active(agents_choice))
        ret['usage'] = usage
        # Flattening of the dictionary
        deep_ret = self.deep_state_flattener(ret)
        logger.debug('[%s] Observation: %s', agent, str(deep_ret))
        return np.array(deep_ret, dtype=np.int64)

    def old_get_obs_space(self, agent):
        """ Returns the observation space. """
        parameters = 0
        parameters += 1                                 # from
        parameters += 1                                 # to
        parameters += 1                                 # time-left
        parameters += len(self.agents[agent].modes)     # ett
        parameters += len(self.agents[agent].modes)     # usage

        lows = [
            0,  # from
            0,  # to
            0,  # time-left
        ]
        lows.extend([-1] * len(self.agents[agent].modes))    # ett
        lows.extend([-1] * len(self.agents[agent].modes))    # usage

        highs = [
            len(self._edges_to_int),                                # from
            len(self._edges_to_int),                                # to
            self._config['scenario_config']['misc']['max_time'],    # time-left
        ]
        highs.extend(                                               # ett
            [self._config['scenario_config']['misc']['max_time']] *
            len(self.agents[agent].modes))
        highs.extend([10] * len(self.agents[agent].modes))          # usage

        print(
            parameters,
            lows, len(lows),
            highs, len(highs),
        )

        return gym.spaces.Box(
            low=np.array(lows), high=np.array(highs), dtype=np.int64)
        # return gym.spaces.Box(low=lows, high=highs, shape=(parameters,), dtype=np.int64)

    def hybrid_get_obs_space(self, agent):
        """ Returns the observation space. """
        parameters = 0
        parameters += 1                                 # from
        parameters += 1                                 # to
        parameters += 1                                 # time-left
        parameters += len(self.agents[agent].modes)     # ett
        parameters += len(self.agents[agent].modes)     # usage

        lows = [
            0,  # from
            0,  # to
            0,  # time-left
        ]
        lows.extend([-1] * len(self.agents[agent].modes))    # ett
        lows.extend([-1] * len(self.agents[agent].modes))    # usage

        highs = [
            len(self._edges_to_int),                                            # from
            len(self._edges_to_int),                                            # to
            self.discrete_time(                                                 # time-left
                self._config['scenario_config']['misc']['max_time']),
        ]
        highs.extend(                                                           # ett
            [self.discrete_time(self._config['scenario_config']['misc']['max_time'])] *
            len(self.agents[agent].modes))
        highs.extend([len(self.agents)] * len(self.agents[agent].modes))        # usage

        print(
            parameters,
            lows, len(lows),
            highs, len(highs),
        )

        return gym.spaces.Box(
            low=np.array(lows), high=np.array(highs), dtype=np.int64)
        # return gym.spaces.Box(low=lows, high=highs, shape=(parameters,), dtype=np.int64)

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

        lows = [
            self.bounding_box['bottom_left_X'],  # from x coord
            self.bounding_box['bottom_left_Y'],  # from y coord
            self.bounding_box['bottom_left_X'],  # to x coord
            self.bounding_box['bottom_left_Y'],  # to y coord
            0,  # time-left
        ]
        lows.extend([-1] * len(self.agents[agent].modes))    # ett
        lows.extend([-1] * len(self.agents[agent].modes))    # usage

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

        logger.info('State space: [%d] - %s - %s', parameters, str(lows), str(highs))
        return gym.spaces.Box(
            low=np.array(lows), high=np.array(highs), dtype=np.float64)

    ################################################################################################

    def get_agents(self):
        """ Returns a list of agents, due to REMOTE, it cannot be an iterarator. """
        keys = [key for key in self.agents.keys()]
        return keys

    def get_expected_arrival(self, agent):
        return self.agents[agent].arrival

    ################################################################################################
