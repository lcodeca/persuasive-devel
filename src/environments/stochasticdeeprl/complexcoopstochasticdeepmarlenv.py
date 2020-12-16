#!/usr/bin/env python3

"""
Stochastic MARL Environment based on ComplexStochasticPersuasiveDeepMARLEnv reward with
added cooperation-related penalty.
"""

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
from environments.stochasticdeeprl.complexstochasticdeepmarlenv import ComplexStochasticPersuasiveDeepMARLEnv

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
    logger.debug('[env_creator] Environment creation: CoopCSPersuasiveDeepMARLEnv')
    return CoopCSPersuasiveDeepMARLEnv(config)

####################################################################################################

#@ray.remote(num_cpus=10, num_gpus=1)
class CoopCSPersuasiveDeepMARLEnv(ComplexStochasticPersuasiveDeepMARLEnv):
    """ Simplified REWARD (with TRAVEL_TIME and COOP), aggregated MODE USAGE, FUTURE DEMAND. """

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

        # While the simulation is ongoing
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
                    logger.warning('Agent %s waited for too long.', str(agent))
                    self.agents[agent].chosen_mode_error = (
                        'Waiting too long [{}]'.format(current_time))
                    self.agents[agent].ett = float('NaN')
                    self.agents[agent].cost = float('NaN')
                    self.dones.add(agent)
                    self.active.remove(agent)
                else:
                    rewards[agent] = 0
                    obs[agent] = self.get_observation(agent)
                    dones[agent] = False
                    infos[agent] = {}

        # in case all the reamining agents WAITED TOO LONG, move forward the simulation
        dones['__all__'] = len(self.dones) == len(self.agents)
        if ongoing_simulation and dones['__all__']:
            logger.info('All the agent are DONE. Finalizing episode...')
            ongoing_simulation = self.simulation.step(
                until_end=dones['__all__'], agents=dones['__all__'])

        # If all are done, compute the reward
        if not ongoing_simulation and dones['__all__']:
            logger.debug('All agents are done, compute the rewards.')
            obs, rewards, dones, infos = self.finalize_episode(dones)
            dones['__all__'] = True

        logger.debug('Observations: %s', str(obs))
        logger.debug('Rewards: %s', str(rewards))
        logger.debug('Dones: %s', str(dones))
        logger.debug('Info: %s', str(infos))
        logger.debug('============================================================================')
        return obs, rewards, dones, infos

    ############################################################################

    def finalize_episode(self, dones):
        """ Gather all the required ifo and aggregate the rewards for the agents. """
        dones['__all__'] = True
        obs, rewards, infos = {}, {}, {}
        ## COMPUTE THE REWARDS
        rewards = self.compute_rewards(self.agents.keys())
        # FINAL STATE observations
        for agent in rewards:
            obs[agent] = self.craft_final_state(agent)
            infos[agent] = self.compute_info_for_agent(agent)
            if agent in self.ext_stats:
                infos[agent]['ext'] = self.ext_stats[agent]
        logger.debug('Observations: %s', str(obs))
        logger.debug('Rewards: %s', str(rewards))
        logger.debug('Dones: %s', str(dones))
        logger.debug('Info: %s', str(infos))
        return obs, rewards, dones, infos

    def compute_rewards(self, agents):
        """ For each agent in the list, return the rewards. """
        rewards = dict()
        goodness_counter = 0
        for agent in agents:
            goodness, reward = self.get_reward(agent)
            goodness_counter += goodness
            rewards[agent] = reward

        if goodness_counter < 1:
            # we cannot have a penalty of 0, given that it's a multiplier, and we deal with
            # negative rewards
            goodness_counter = 1

        # if all the agents are good, this penalty is 1 and it disappear.
        penalty = 1.0 / (goodness_counter / len(agents))
        for agent in agents:
            logger.debug('Agent %s: reward %f - penalty %f - outcome %f',
                         agent, rewards[agent], penalty, rewards[agent] * penalty)
            rewards[agent] = rewards[agent] * penalty
        logger.debug('Goodness: %d (%d)', goodness_counter, len(agents))
        return rewards

    def get_reward(self, agent):
        """ Return the reward for a given agent without the cooperation counted. """

        arrival = self.simulation.get_arrival(agent, default=None)
        #### ERRORS
        if arrival is None:
            logger.warning('Reward: Error for agent %s.', agent)
            # the maximum penalty is set as the slowest travel time,
            #   while starting when is too late.
            max_travel_time = self.agents[agent].max_travel_time
            max_travel_time /= self._config['agent_init']['travel-slots-min'] # slotted time
            return 0, 0 - (max_travel_time * 2) * self.agents[agent].late_weight

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
            return 0, 0 - (travel_time + penalty) * self.agents[agent].late_weight

        arrival_buffer = (
            self.agents[agent].arrival - (self._config['agent_init']['arrival-slots-min'] * 60))

        #### TOO EARLY
        if arrival_buffer > arrival:
            logger.debug('Reward: Agent %s arrived too early.', agent)
            penalty = self.agents[agent].arrival - arrival
            penalty /= 60 # in minutes
            penalty /= self._config['agent_init']['travel-slots-min'] # slotted time
            return 1, 0 - (travel_time + penalty) * self.agents[agent].waiting_weight

        #### ON TIME
        logger.info('Reward: Agent %s arrived on time.', agent)
        return 1, 1 - travel_time

    ################################################################################################
