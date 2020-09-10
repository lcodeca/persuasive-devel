#!/usr/bin/env python3

"""
    Stand-Alone Trainer for Q-Learning with a probability distribution for the epsilon-action
    based on QLearningTrainer and EGreedyQLearningPolicy.
"""
from copy import deepcopy
import logging
from pprint import pformat

from utils.qtable import QTable

from learning.ql.qlearningstandalonetrainer import QLearningTrainer, EGreedyQLearningPolicy

####################################################################################################

DEBUGGER = False
PROFILER = False

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)

####################################################################################################
#                                             TRAINER
####################################################################################################

class ProbabilityDistributionQLearningTrainer(QLearningTrainer):
    """
        Stand-Alone Trainer for Q-Learning with a probability distribution for the epsilon-action
        based on QLearningTrainer and EGreedyQLearningPolicy.
    """

    def _initialize_policies(self, config):
        self.policies = dict()
        for agent, parameters in config['multiagent']['policies'].items():
            _, obs_space, action_space, add_cfg = parameters
            self.policies[agent] = ProbabilityDistributionEGreedyQLearningPolicy(
                obs_space, action_space, add_cfg)

####################################################################################################
#                                             POLICY
####################################################################################################

class ProbabilityDistributionEGreedyQLearningPolicy(EGreedyQLearningPolicy):
    """
        Stand-Alone Trainer for Q-Learning with a probability distribution for the epsilon-action
        based on QLearningTrainer and EGreedyQLearningPolicy.
    """

    def __init__(self, observation_space, action_space, config):
        """
        Example of a config = {
            'actions': {0, 1, 2},
            'actions-distribution': [0.8, 0.1, 0.1],
            'alpha': 0.1,
            'decay': 0.9,
            'epsilon': 0.1,
            'gamma': 0.6,
            'seed': 42,
            'init': 0.0,
        }
        """
        EGreedyQLearningPolicy.__init__(self, observation_space, action_space, config)
        # Additional Parameters
        self.cumulative_probability = list()
        cumul = 0.0
        for probability in config['actions-distribution']:
            cumul += probability
            self.cumulative_probability.append(cumul)
        logger.debug('Cumulative probability: %s', pformat(self.cumulative_probability))

    def _get_action_from_distribution(self, value):
        # self.action_space.sample()
        for pos, prob in enumerate(self.cumulative_probability):
            if value <= prob:
                return pos
        raise Exception(
            '_get_action_from_distribution: {} {}'.format(value, self.cumulative_probability))

    def compute_action(self, state):
        # Epsilon-Greedy Implementation
        if DEBUGGER:
            logger.debug('Observation: %s', pformat(state))
        action = None

        rnd = self.rndgen.uniform(0, 1)
        logger.debug('Random: %f - Epsilon: %f - value %s',
                     rnd, self.epsilon, str(rnd < self.epsilon))
        if rnd < self.epsilon:
            # Explore action space
            rnd = self.rndgen.uniform(0, 1)
            action = self._get_action_from_distribution(rnd)
            logger.debug('Rnd value: %f - Action: %d', rnd, action)
        else:
            # Exploit learned values
            action = self.qtable.argmax(state)
            if DEBUGGER:
                logger.debug('State: %s --> action: %s', pformat(self.qtable[state]), str(action))

        self.stats['actions'].append(action)
        self.qtable_state_action_counter[state][action] += 1
        return action

    def get_internal_state(self):
        """ Returns a dict containing the internal state of the policy. """
        state = super().get_internal_state()
        state['cumulative_probability'] = self.cumulative_probability
        return state

    def set_internal_state(self, internal_state):
        """ Sets the internal state of the policy from a dict. """
        self.cumulative_probability = internal_state.pop('cumulative_probability', [])
        super().set_internal_state(internal_state)
