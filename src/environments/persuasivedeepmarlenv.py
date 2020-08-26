#!/usr/bin/env python3

""" Initial Agents Cooperation MARL Environment based on PersuasiveMultiAgentEnv """

import logging
import os
import sys

from collections import defaultdict
from copy import deepcopy
from pprint import pformat

import numpy as np

import gym
import ray

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
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)

####################################################################################################

def env_creator(config):
    """ Environment creator used in the environment registration. """
    logger.debug('[env_creator] Environment creation: PersuasiveDeepMARLEnv')
    return PersuasiveDeepMARLEnv(config)

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

#@ray.remote(num_cpus=10, num_gpus=1)
class PersuasiveDeepMARLEnv(PersuasiveMultiAgentEnv):
    """
    Initial implementation of Late Reward and Agents Cooperation based on the
    PersuasiveMultiAgentEnv, explicitly for deep learning implementation. """

    def __init__(self, config):
        """ Initialize the environment. """
        super().__init__(config)
        self.episode_snapshot = AgentsHistory()

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
        logger.debug('Usage: %d / %d * 100 / 10 = %d (rounded).', active, len(self.agents), ret)
        return ret

    ################################################################################################

    def reset(self):
        """ Resets the env and returns observations from ready agents. """
        initial_obs = super().reset()
        self.episode_snapshot = AgentsHistory()
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
        logger.debug('========================================================')
        logger.debug('Snapshot: \n%s', str(self.episode_snapshot))
        logger.debug('========================================================')
        return super().step(action_dict)

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
            raise Exception('{} \n {}'.format(
                self.compute_info_for_agent(agent), str(self.agents[agent])))
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
            state['from'],
            state['to'],
            state['time-left']
        ]
        deep.extend(state['ett'])
        deep.extend(state['usage'])
        return deepcopy(deep)

    def craft_final_state(self, agent):
        final_state = super().craft_final_state(agent)
        final_state['usage'] = np.array([-1 for _ in self.agents[agent].modes])
        return self.deep_state_flattener(final_state)

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
        # Flattening of the dictionary
        deep_ret = self.deep_state_flattener(ret)
        logger.debug('Observation: %s', str(deep_ret))
        return np.array(deep_ret, dtype=np.int64)

    def get_obs_space(self, agent):
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

    ################################################################################################

    def get_agents(self):
        """ Returns a list of agents, due to REMOTE, it cannot be an iterarator. """
        keys = [key for key in self.agents.keys()]
        return keys

    def get_expected_arrival(self, agent):
        return self.agents[agent].arrival

    ################################################################################################
