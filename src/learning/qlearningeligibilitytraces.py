#!/usr/bin/env python3

"""
    Stand-Alone Trainer for Q-Learning with Eligibility Traces Trainer based on QLearningTrainer
"""
from copy import deepcopy
import logging
from pprint import pformat

from utils.qtable import QTable

from learning.qlearningstandalonetrainer import QLearningTrainer, EGreedyQLearningPolicy

####################################################################################################

DEBUGGER = False
PROFILER = False

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

####################################################################################################
#                                             TRAINER
####################################################################################################

class QLearningEligibilityTracesTrainer(QLearningTrainer):
    """
    See:
        https://towardsdatascience.com/eligibility-traces-in-reinforcement-learning-a6b458c019d6
    """

    def _init(self, config, env_creator):
        """ Q-Learning Trainer init. """
        LOGGER.debug('QLearningEligibilityTracesTrainer:_init() MARL Environment Creation..')
        self._latest_checkpoint = ''
        self.env = env_creator(config['env_config'])
        self.policies = dict()
        for agent, parameters in config['multiagent']['policies'].items():
            _, obs_space, action_space, add_cfg = parameters
            self.policies[agent] = EGreedyQLearningEligibilityTracesPolicy(
                obs_space, action_space, add_cfg)

    def on_episode_start(self):
        for agent in self.policies.values():
            agent.reset_eligibility_trace()

####################################################################################################
#                                             POLICY
####################################################################################################

class EGreedyQLearningEligibilityTracesPolicy(EGreedyQLearningPolicy):
    """
    See:
        https://towardsdatascience.com/eligibility-traces-in-reinforcement-learning-a6b458c019d6
    """

    def __init__(self, observation_space, action_space, config):
        """
        Example of a config = {
            'actions': {0, 1, 2},
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
        self.decay = deepcopy(config['decay'])
        self.eligibility_trace = QTable(self.set_of_actions, default=0.0)

    def reset_eligibility_trace(self):
        """ Forcefully reset the eligibility trace """
        del self.eligibility_trace
        self.eligibility_trace = QTable(self.set_of_actions, default=0.0)

    def learn(self, sample):
        """
        Q-Learning with Eligibility Traces implementation

        See:
            https://en.wikipedia.org/wiki/Q-learning#Algorithm
            https://stackoverflow.com/questions/40862578/how-to-understand-watkinss-q%CE%BB-learning-algorithm-in-suttonbartos-rl-book

        Given a sample = {
                        'old_state': states[agent],
                        'action': actions[agent],
                        'next_state': next_states[agent],
                        'reward': rewards[agent],
                        'info': infos[agent],
                    }
        """
        if DEBUGGER:
            LOGGER.debug('Learning sample \n%s', pformat(sample))
            LOGGER.debug('Old State \n%s', pformat(self.qtable[sample['old_state']]))
            LOGGER.debug('Next State \n%s', pformat(self.qtable[sample['next_state']]))

        # in case the action chosen was a greedy one
        best_actions = self.qtable.maxactions(sample['old_state'])
        LOGGER.debug('Q-Learning: best actions = %s, action taken = %d',
                     str(best_actions), sample['action'])
        if sample['action'] not in best_actions:
            LOGGER.debug('Q-Learning: the eligibility trace will be reset.')
        best_action = list(best_actions)[0]

        # compute the error
        error = (
            sample['reward'] +
            (self.gamma * self.qtable[sample['next_state']][best_action]) -
            self.qtable[sample['old_state']][sample['action']])
        LOGGER.debug(
            'Q-Learning: error = %.2f + %.2f * %.2f - %.2f',
            sample['reward'], self.gamma, self.qtable[sample['next_state']][sample['action']],
            self.qtable[sample['old_state']][best_action])
        LOGGER.debug('Q-Learning: error = %.2f', error)

        # increment the eligibility trace
        self.eligibility_trace[sample['old_state']][sample['action']] += 1.0

        for key, values in self.eligibility_trace.get_items():
            LOGGER.debug('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            LOGGER.debug('Serialized key %s', key)
            state = self.eligibility_trace.get_state_from_serialized_key(key)
            LOGGER.debug('State %s', pformat(state))
            LOGGER.debug('Values %s', pformat(values))
            for action in values:
                LOGGER.debug('===========================> Action %d <===========================',
                             action)
                # adjust q-values
                old_qvalue = self.qtable[state][action]
                new_qvalue = old_qvalue + self.alpha * error * self.eligibility_trace[state][action]
                LOGGER.debug('q-value = %.2f + %.2f * %.2f * %.2f',
                             old_qvalue, self.alpha, error, self.eligibility_trace[state][action])
                LOGGER.debug('q-values: old = %f, new = %f', old_qvalue, new_qvalue)
                self.qtable[state][action] = new_qvalue
                # update eligibility traces
                if sample['action'] in best_actions:
                    # update the trace
                    LOGGER.debug('old e-trace %f', self.eligibility_trace[state][action])
                    self.eligibility_trace[state][action] *= self.gamma * self.decay
                    LOGGER.debug('new e-trace %f', self.eligibility_trace[state][action])
                else:
                    # reset the trace because the action chosen was random
                    LOGGER.debug('Q-Learning: resetting trace.')
                    self.eligibility_trace[state][action] = 0.0
                LOGGER.debug('==================================================================')
            LOGGER.debug('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        if DEBUGGER:
            LOGGER.debug('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            LOGGER.debug('Q-Learning: eligibility traces \n%s', str(self.eligibility_trace))
            LOGGER.debug('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        # STATS
        self.stats['rewards'].append(sample['reward'])
        self.qtable_state_action_reward[sample['old_state']][sample['action']].append(
            sample['reward'])
        self.stats['sequence'].append(
            (sample['old_state'], sample['action'], sample['next_state'], sample['reward']))
        if sample['info']:
            sample['info']['from-state'] = sample['old_state']
            self.stats['info'].append(sample['info'])
