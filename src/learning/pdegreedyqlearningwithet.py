#!/usr/bin/env python3

"""
    Stand-Alone Trainer for Q-Learning with a probability distribution for the epsilon-action
    based on both ProbabilityDistributionEGreedyQLearningPolicy and
    EGreedyQLearningEligibilityTracesPolicy.
"""
from copy import deepcopy
import logging
from pprint import pformat

from utils.qtable import QTable

from learning.probegreedyqlearning import (ProbabilityDistributionQLearningTrainer,
                                           ProbabilityDistributionEGreedyQLearningPolicy)
from learning.qlearningeligibilitytraces import (QLearningEligibilityTracesTrainer,
                                                 EGreedyQLearningEligibilityTracesPolicy)

####################################################################################################

DEBUGGER = False
PROFILER = False

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)

####################################################################################################
#                                             TRAINER
####################################################################################################

class PDEGreedyQLearningETTrainer(
        QLearningEligibilityTracesTrainer, ProbabilityDistributionQLearningTrainer):
    """
        Stand-Alone Trainer for Q-Learning with a probability distribution for the epsilon-action
        and eligibility traces.
    """

    def _initialize_policies(self, config):
        self.policies = dict()
        for agent, parameters in config['multiagent']['policies'].items():
            _, obs_space, action_space, add_cfg = parameters
            self.policies[agent] = PDEGreedyQLearningETPolicy(
                obs_space, action_space, add_cfg)

####################################################################################################
#                                             POLICY
####################################################################################################

class PDEGreedyQLearningETPolicy(
        EGreedyQLearningEligibilityTracesPolicy, ProbabilityDistributionEGreedyQLearningPolicy):
    """
        Policy for Q-Learning with a probability distribution for the epsilon-action
        and eligibility traces.
    """

    def __init__(self, observation_space, action_space, config):
        """
        Example of a config = {
            'actions': {0, 1, 2},
            'actions-distribution': [0.8, 0.1, 0.1],
            'alpha': 0.1,
            'epsilon': 0.1,
            'gamma': 0.6,
            'seed': 42,
            'init': 0.0,
        }
        """
        EGreedyQLearningEligibilityTracesPolicy.__init__(
            self, observation_space, action_space, config)
        ProbabilityDistributionEGreedyQLearningPolicy.__init__(
            self, observation_space, action_space, config)
