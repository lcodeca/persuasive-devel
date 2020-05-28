#!/usr/bin/env python3

""" MARL Environment with late arrival penalty based on PersuasiveMultiAgentEnv. """

import logging
import os
import sys

import numpy as np

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
    LOGGER.debug('[env_creator] Environment creation: LateRewardMultiAgentEnv')
    return LateRewardMultiAgentEnv(config)

####################################################################################################

class LateRewardMultiAgentEnv(PersuasiveMultiAgentEnv):
    """ Initial implementation of Agents Cooperation based on the PersuasiveMultiAgentEnv. """

    ################################################################################################

    def get_reward(self, agent):
        """ Return the reward for a given agent. """
        if not self.agents[agent].chosen_mode:
            LOGGER.warning('Agent %s mode error: "%s"', agent, self.agents[agent].chosen_mode_error)
            return 0 - int(self.simulation.get_penalty_time())

        journey_time = self.simulation.get_duration(agent)
        if np.isnan(journey_time):
            ## This should never happen.
            ## If it does, there is a bug/issue with the SUMO/MARL environment interaction.
            raise Exception('{} \n {}'.format(
                self.compute_info_for_agent(agent), str(self.agents[agent])))
        LOGGER.debug(' Agent: %s, journey: %s', agent, str(journey_time))
        arrival = self.simulation.get_arrival(agent,
                                              default=self.simulation.get_penalty_time())
        LOGGER.debug(' Agent: %s, arrival: %s', agent, str(arrival))

        # REWARD = journey time * mode weight + ....
        reward = journey_time * self.agents[agent].modes[self.agents[agent].chosen_mode]
        if self.agents[agent].arrival < arrival:
            ## agent arrived too late
            late_time = arrival - self.agents[agent].arrival
            reward += late_time * self.agents[agent].late_weight
            LOGGER.debug('Agent: %s, arrival: %s, wanted arrival: %s, late: %s',
                         agent, str(arrival), str(self.agents[agent].arrival), str(late_time))
        elif self.agents[agent].arrival > arrival:
            ## agent arrived too early
            waiting_time = self.agents[agent].arrival - arrival
            reward += waiting_time * self.agents[agent].waiting_weight
            LOGGER.debug('Agent: %s, duration: %s, waiting: %s, wanted arrival: %s',
                         agent, str(journey_time), str(waiting_time), str(arrival))
        else:
            LOGGER.debug('Agent: %s it is perfectly on time!', agent)

        return int(0 - (reward))
