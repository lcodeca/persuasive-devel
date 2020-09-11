#!/usr/bin/env python3

"""
    Stand-Alone Trainer for Q-Learning Trainer based on RLLIB
    See:
        https://ray.readthedocs.io/en/latest/rllib-concepts.html#trainers
        https://ray.readthedocs.io/en/latest/tune-usage.html#trainable-api
        https://ray.readthedocs.io/en/latest/tune-package-ref.html#ray.tune.Trainable

        https://ray.readthedocs.io/en/latest/rllib-concepts.html#policies
        https://github.com/keras-rl/keras-rl/blob/master/rl/policy.py#L141
        https://www.geeksforgeeks.org/q-learning-in-python/

"""
import collections
from copy import deepcopy
import cProfile
from datetime import datetime
import io
import logging
import os
import pstats
import sys
from pprint import pformat

import numpy as np
from numpy.random import RandomState

import dill

from ray.rllib.agents.trainer import Trainer, with_common_config
from ray.rllib.policy import Policy

from utils.qtable import QTable
from utils.logger import set_logging

# """ Import SUMO library """
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
    from traci.exceptions import TraCIException, FatalTraCIError
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

####################################################################################################

DEBUGGER = False
PROFILER = False

logger = set_logging(__name__)

####################################################################################################
#                                             TRAINER
####################################################################################################

class QLearningTrainer(Trainer):
    """
    See:
        Before: https://github.com/ray-project/ray/blob/releases/0.7.6/python/ray/tune/trainable.py
        After: https://github.com/ray-project/ray/blob/releases/0.7.6/rllib/agents/trainer.py#L303
    """

    # DEFAULT Configuration
    _default_config = with_common_config({})

    def _init(self, config, env_creator):
        """ Q-Learning Trainer init. """
        logger.debug('QLearningTrainer:_init() MARL Environment Creation..')
        self._latest_checkpoint = ''
        self.env = env_creator(config['env_config'])
        self._initialize_policies(config)

    def _initialize_policies(self, config):
        self.policies = dict()
        for agent, parameters in config['multiagent']['policies'].items():
            _, obs_space, action_space, add_cfg = parameters
            self.policies[agent] = EGreedyQLearningPolicy(obs_space, action_space, add_cfg)

    # def _setup(self, config):
    #     """ Subclasses should override this for custom initialization.

    #     Args:
    #         config (dict): Hyperparameters and other configs given.
    #             Copy of `self.config`.
    #     """
    #     super()._setup(config)

    def _train(self):
        """ Subclasses should override this to implement train().

        The return value will be automatically passed to the loggers. Users
        can also return `tune.result.DONE` or `tune.result.SHOULD_CHECKPOINT`
        as a key to manually trigger termination or checkpointing of this
        trial. Note that manual checkpointing only works when subclassing
        Trainables.

        Returns:
            A dict that describes training progress.
        """

        ## ========================              PROFILER              ======================== ##
        if PROFILER:
            profiler = cProfile.Profile()
            profiler.enable()
        ## ========================              PROFILER              ======================== ##

        logger.debug('QLearningTrainer:_train()')

        gtt_by_episode = list()
        sumo_steps_per_episode = list()
        rewards_by_episode = list()
        elapesed_time_by_episode = list()
        policies_aggregated_by_episode = collections.defaultdict(dict)
        def aggregate_policies(policy, internal_state, statistics):
            """ Equivalent of a moving average between episodes. """
            if DEBUGGER:
                logger.debug('%s', pformat(policy, compact=True))
            if policy:
                # aggregate
                policy['state'] = internal_state
                policy['stats']['sequence'].append(statistics['sequence'])
                policy['stats']['info'].append(statistics['info'])
                policy['stats']['actions_this_episode'] += (
                    (statistics['actions_this_episode'] - policy['stats']['actions_this_episode'])
                    / policy['episodes'])
                policy['stats']['agent_reward'] += (
                    (statistics['agent_reward'] - policy['stats']['agent_reward'])
                    / policy['episodes'])
                policy['stats']['agent_reward_max'] += (
                    (statistics['agent_reward_max'] - policy['stats']['agent_reward_max'])
                    / policy['episodes'])
                policy['stats']['agent_reward_mean'] += (
                    (statistics['agent_reward_mean'] - policy['stats']['agent_reward_mean'])
                    / policy['episodes'])
                policy['stats']['agent_reward_min'] += (
                    (statistics['agent_reward_min'] - policy['stats']['agent_reward_min'])
                    / policy['episodes'])
                policy['episodes'] += 1
            else:
                policy = {
                    'episodes': 1,
                    'state': internal_state,
                    'stats': {
                        'actions_this_episode': statistics['actions_this_episode'],
                        'agent_reward': statistics['agent_reward'],
                        'agent_reward_max': statistics['agent_reward_max'],
                        'agent_reward_mean': statistics['agent_reward_mean'],
                        'agent_reward_min': statistics['agent_reward_min'],
                        'sequence': [statistics['sequence']],
                        'info': [statistics['info']]
                    },
                }
            if DEBUGGER:
                logger.debug('%s', pformat(policy, compact=True))
            return policy

        learning_steps = 0
        for episode in range(self.config['rollout_fragment_length']):
            before = datetime.now()
            logger.info('=======================> Episode # %4d <=======================', episode)
            max_retry = 10
            steps = 0
            while max_retry:
                try:
                    # callback
                    self.on_episode_start()

                    # start from the beginning
                    states = self.env.reset()

                    steps = 0

                    cumul_rewards_by_agent = collections.defaultdict(int)
                    dones = {'__all__': False,}

                    latest_state_by_agent = {}
                    for agent, state in states.items():
                        latest_state_by_agent[agent] = state

                    latest_action_by_agent = {}

                    # until all the agents are not done
                    while not dones['__all__']:
                        # callback
                        self.on_episode_step()

                        if states:
                            # Possibility due to the decoupling of the sumo environment and the
                            # learning environment, it's possible that not all of the agents
                            # are done, but no agent is active atm and the sumo environment
                            # needs to keep moving forward nonetheless.
                            steps += 1
                        if DEBUGGER:
                            logger.debug('State: %s', pformat(states))
                        actions = {}
                        for agent, state in states.items():
                            actions[agent] = self.policies[agent].compute_action(state)
                            latest_action_by_agent[agent] = actions[agent]
                            logger.debug('Agent %s selected action %d', agent, actions[agent])
                        if DEBUGGER:
                            logger.debug('Actions: %s', pformat(actions))

                        logger.debug('STEP!')
                        next_states, rewards, dones, infos = self.env.step(actions)

                        if DEBUGGER:
                            logger.debug('Observations: %s', pformat(next_states))
                            logger.debug('Rewards: %s', pformat(rewards))
                            logger.debug('Dones: %s', pformat(dones))
                            logger.debug('Info: %s', pformat(infos))

                        for agent in next_states:
                            if agent not in latest_state_by_agent:
                                ## this agent has just been inserted in the simulation
                                continue
                            else:
                                ## we have something to learn here
                                sample = {
                                    'old_state': latest_state_by_agent[agent],
                                    'action': latest_action_by_agent[agent],
                                    'next_state': next_states[agent],
                                    'reward': rewards[agent],
                                    'info': infos[agent],
                                }
                                if DEBUGGER:
                                    logger.debug('Learning sample for agent %s: \n%s',
                                                 agent, pformat(sample))
                                self.policies[agent].learn(sample)
                                cumul_rewards_by_agent[agent] += rewards[agent]

                        states = next_states
                        for agent, state in states.items():
                            latest_state_by_agent[agent] = state
                    # very dirty, but it works :)
                    break
                except TraCIException as excpt:
                    max_retry -= 1
                    logger.critical('SUMO failed with TraCIException: %s', pformat(excpt))
                except FatalTraCIError as error:
                    max_retry -= 1
                    logger.critical('SUMO failed with FatalTraCIError: %s', pformat(error))

            # Gathering metrics at the end of the episode
            if max_retry:
                logger.debug('Learning steps this iteration: %d', steps)
                learning_steps += steps
                gtt_by_episode.append(self.env.simulation.get_global_travel_time())
                sumo_steps_per_episode.append(self.env.simulation.get_sumo_steps())
                rewards_by_episode.append(cumul_rewards_by_agent)
                for agent, policy in self.policies.items():
                    logger.debug('Collecting stats from agent: %s', agent)
                    policies_aggregated_by_episode[agent] = aggregate_policies(
                        policies_aggregated_by_episode[agent],
                        policy.get_internal_state(),
                        policy.get_stats_and_reset())
            delta = datetime.now() - before
            logger.info('=======================> %s <=======================', str(delta))
            elapesed_time_by_episode.append(delta.total_seconds())
            # callback
            self.on_episode_end()

        # Metrics gathering, averaged by number of episodes.
        aggregated_rewards_per_episode = list()
        averaged_rewards_by_agent = collections.defaultdict(list)
        for episode in rewards_by_episode:
            episode_reward = list()
            for agent, reward in episode.items():
                averaged_rewards_by_agent[agent].append(reward)
                episode_reward.append(reward)
            aggregated_rewards_per_episode.append(np.mean(episode_reward))
        for agent, values in averaged_rewards_by_agent.items():
            averaged_rewards_by_agent[agent] = np.mean(values)

        for agent, policy in policies_aggregated_by_episode.items():
            if DEBUGGER:
                logger.debug('[%s] BEFORE \n%s', agent, pformat(policy, compact=True))

            policy['qtable'] = collections.defaultdict(dict)
            policy['max-qvalue'] = dict()
            policy['best-action'] = dict()
            for item in dill.loads(policy['state']['qtable']).get_flattened_dict():
                _state, _action, _value = item
                policy['qtable'][_state][str(_action)] = _value
                if _state in policy['max-qvalue']:
                    policy['max-qvalue'][_state] = max(
                        policy['max-qvalue'][_state], _value)
                    policy['best-action'][_state] = max(
                        policy['best-action'][_state], (_value, _action))
                else:
                    policy['max-qvalue'][_state] = _value
                    policy['best-action'][_state] = (_value, _action)
            for _state, value in policy['best-action'].items():
                _, _action = value
                policy['best-action'][_state] = _action

            policy['state-action-counter'] = collections.defaultdict(dict)
            for item in policy['state']['qtable_state_action_counter'].get_flattened_dict():
                _state, _action, _value = item
                policy['state-action-counter'][_state][str(_action)] = _value

            policy['state-action-reward-mean'] = collections.defaultdict(dict)
            for item in policy['state']['qtable_state_action_reward'].get_flattened_dict():
                _state, _action, _value = item
                policy['state-action-reward-mean'][_state][str(_action)] = np.mean(_value)

            if DEBUGGER:
                logger.debug('[%s] AFTER \n%s', agent, pformat(policy, compact=True))

        ## ========================              PROFILER              ======================== ##
        if PROFILER:
            profiler.disable()
            results = io.StringIO()
            pstats.Stats(profiler, stream=results).sort_stats('cumulative').print_stats(25)
            print(results.getvalue())
        ## ========================              PROFILER              ======================== ##

        return {
            'episode_reward_mean': np.mean(aggregated_rewards_per_episode),
            'episodes_this_iter': self.config['rollout_fragment_length'],
            'episode_elapsed_time_mean': np.mean(elapesed_time_by_episode),
            'timesteps_this_iter': learning_steps,
            'sumo_steps_this_iter': np.mean(sumo_steps_per_episode),
            'environment_steps_this_iter': self.env.get_environment_steps(),
            'rewards': averaged_rewards_by_agent,
            'policies': policies_aggregated_by_episode,
            'episode_gtt_mean': np.mean(gtt_by_episode),
            'episode_gtt_max': max(gtt_by_episode, default=None),
            'episode_gtt_min': min(gtt_by_episode, default=None),
        }

    def _save(self, tmp_checkpoint_dir):
        """ Subclasses should override this to implement ``save()``. """
        logger.debug('Checkpoint directory: %s', tmp_checkpoint_dir)
        checkpoint = dict()
        for key, item in self.policies.items():
            logger.debug('Policy[%s]: \n%s', key, item)
            checkpoint[key] = item.get_internal_state()
        return checkpoint

    def _restore(self, checkpoint):
        """ Subclasses should override this to implement restore(). """
        for key, item in checkpoint.items():
            logger.debug('Checkpoint[%s]: \n%s', key, item)
            if 'tune_checkpoint_path' in key:
                self._latest_checkpoint = checkpoint['tune_checkpoint_path']
            else:
                self.policies[key].set_internal_state(item)

    def _log_result(self, result):
        """ Subclasses can optionally override this to customize logging.

        Args:
            result (dict): Training result returned by _train().
        """
        # See: https://github.com/ray-project/ray/blob/master/python/ray/tune/logger.py#L177

        # callback
        self.on_train_result(result)

        self._result_logger.on_result(result)

    ################################################################################################
    #                                         CALLBACKS
    ################################################################################################

    def on_episode_start(self):
        pass

    def on_episode_step(self):
        pass

    def on_episode_end(self):
        pass

    def on_train_result(self, result):
        pass

####################################################################################################
#                                             POLICY
####################################################################################################

class EGreedyQLearningPolicy(Policy):
    """
    Unable to implement:
        https://ray.readthedocs.io/en/latest/rllib-concepts.html#policies
        https://github.com/ray-project/ray/blob/master/rllib/policy/policy.py
    this policy is not distributed among the RAY workers.
    """

    def __init__(self, observation_space, action_space, config):
        """
        Example of a config = {
            'actions': {0, 1, 2},
            'alpha': 0.1,
            'epsilon': 0.1,
            'gamma': 0.6,
            'seed': 42,
            'init': 0.0,
        }
        """
        Policy.__init__(self, observation_space, action_space, config)
        # Parameters
        self.set_of_actions = deepcopy(config['actions'])
        self.alpha = deepcopy(config['alpha'])
        self.gamma = deepcopy(config['gamma'])
        self.epsilon = deepcopy(config['epsilon'])
        self.qtable = QTable(self.set_of_actions, default=config['init'], seed=config['seed'])
        self.qtable_state_action_counter = QTable(self.set_of_actions, default=0)
        self.qtable_state_action_reward = QTable(self.set_of_actions, default=list())
        # self.qtable_new_state_action_total_reward = QTable(self.set_of_actions, default=list())
        self.rndgen = RandomState(config['seed'])
        # Logging
        self.stats = dict()
        self._reset_stats_values()

    def _reset_stats_values(self):
        """ Reset all the stats metrics used for logging. """
        self.stats['actions'] = list()
        self.stats['rewards'] = list()
        self.stats['sequence'] = list()
        self.stats['info'] = list()

    def get_stats_and_reset(self):
        """ Returns all the stats metrics used for logging (and reset them). """
        if DEBUGGER:
            logger.debug('Stats: \n%s', pformat(self.stats))
        statscp = deepcopy(self.stats)
        self._reset_stats_values()
        statscp['actions_this_episode'] = len(statscp['actions'])
        statscp['agent_reward'] = statscp['rewards'][-1]
        statscp['agent_reward_mean'] = np.mean(statscp['rewards'])
        statscp['agent_reward_min'] = min(statscp['rewards'], default=None)
        statscp['agent_reward_max'] = max(statscp['rewards'], default=None)
        return statscp

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
            action = self.action_space.sample()
            logger.debug('Random (%f) action: %d', rnd, action)
        else:
            # Exploit learned values
            action = self.qtable.argmax(state)
            if DEBUGGER:
                logger.debug('State: %s --> action: %s', pformat(self.qtable[state]), str(action))

        self.stats['actions'].append(action)
        self.qtable_state_action_counter[state][action] += 1
        return action

    def learn(self, sample):
        """
        Q-Learning implementation

        See: https://en.wikipedia.org/wiki/Q-learning#Algorithm

        Given a sample = {
                        'old_state': states[agent],
                        'action': actions[agent],
                        'next_state': next_states[agent],
                        'reward': rewards[agent],
                        'info': infos[agent],
                    }
        """
        if DEBUGGER:
            logger.debug('Learning sample \n%s', pformat(sample))
            logger.debug('Old State \n%s', pformat(self.qtable[sample['old_state']]))
            logger.debug('Next State \n%s', pformat(self.qtable[sample['next_state']]))
        old_value = self.qtable[sample['old_state']][sample['action']]
        next_max = self.qtable.max(sample['next_state'])
        new_value = old_value + self.alpha * (sample['reward'] + self.gamma * next_max - old_value)
        logger.debug('%f = %f + %f * (%f + %f * %f - %f)',
                     new_value, old_value, self.alpha, sample['reward'], self.gamma,
                     next_max, old_value)
        self.qtable[sample['old_state']][sample['action']] = new_value
        logger.debug('Q-Learning: old = %f, new = %f', old_value, new_value)

        self.stats['rewards'].append(sample['reward'])
        self.qtable_state_action_reward[sample['old_state']][sample['action']].append(
            sample['reward'])
        self.stats['sequence'].append(
            (sample['old_state'], sample['action'], sample['next_state'], sample['reward']))
        if sample['info']:
            sample['info']['from-state'] = sample['old_state']
            self.stats['info'].append(sample['info'])

    def get_internal_state(self):
        """ Returns a dict containing the internal state of the policy. """
        state = {
            'qtable': dill.dumps(self.qtable),
            'qtable_state_action_counter': self.qtable_state_action_counter,
            'qtable_state_action_reward': self.qtable_state_action_reward,
            'rndgen': self.rndgen,
            'actions': self.set_of_actions,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
        }
        return state

    def set_internal_state(self, internal_state):
        """ Sets the internal state of the policy from a dict. """
        self.qtable = dill.loads(internal_state['qtable'])
        self.qtable_state_action_counter = internal_state['qtable_state_action_counter']
        self.qtable_state_action_reward = internal_state['qtable_state_action_reward']
        self.rndgen = internal_state['rndgen']
        self.set_of_actions = internal_state['actions']
        self.alpha = internal_state['alpha']
        self.gamma = internal_state['gamma']
        self.epsilon = internal_state['epsilon']

    def compute_actions(self,
                        obs_batch,
                        state_batches,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        """ Compute actions for the current policy.

        Arguments:
            obs_batch (np.ndarray): batch of observations
            state_batches (list): list of RNN state input batches, if any
            prev_action_batch (np.ndarray): batch of previous action values
            prev_reward_batch (np.ndarray): batch of previous rewards
            info_batch (info): batch of info objects
            episodes (list): MultiAgentEpisode for each obs in obs_batch.
                This provides access to all of the internal episode state,
                which may be useful for model-based or multiagent algorithms.
            kwargs: forward compatibility placeholder

        Returns:
            actions (np.ndarray): batch of output actions, with shape like
                [BATCH_SIZE, ACTION_SHAPE].
            state_outs (list): list of RNN state output batches, if any, with
                shape like [STATE_SIZE, BATCH_SIZE].
            info (dict): dictionary of extra feature batches, if any, with
                shape like {"f1": [BATCH_SIZE, ...], "f2": [BATCH_SIZE, ...]}.
        """
        return [], [], {}

####################################################################################################
#                                      TESTING "TRAINER"
####################################################################################################

class QLearningTester(QLearningTrainer):
    """ Testing environment for a QLearningTrainer policy """

    def _initialize_policies(self, config):
        self.policies = dict()
        for agent, parameters in config['multiagent']['policies'].items():
            _, obs_space, action_space, add_cfg = parameters
            self.policies[agent] = QLearningTestingPolicy(obs_space, action_space, add_cfg)


####################################################################################################
#                                       TESTING POLICY
####################################################################################################

class QLearningTestingPolicy(EGreedyQLearningPolicy):
    """ Testing policy """

    def __init__(self, observation_space, action_space, config):
        """
        Example of a config = {
            'actions': {0, 1, 2},
            'seed': 42,
            'init': 0.0,
        }
        """
        Policy.__init__(self, observation_space, action_space, config)
        # Parameters
        self.set_of_actions = deepcopy(config['actions'])
        self.qtable = QTable(self.set_of_actions, default=config['init'], seed=config['seed'])
        self.qtable_state_action_counter = QTable(self.set_of_actions, default=0)
        self.qtable_state_action_reward = QTable(self.set_of_actions, default=list())
        self.rndgen = RandomState(config['seed'])
        # Logging
        self.stats = dict()
        self._reset_stats_values()

    def compute_action(self, state):
        # Epsilon-Greedy Implementation
        if DEBUGGER:
            logger.debug('Observation: %s', pformat(state))

        # Exploit learned values
        action = self.qtable.argmax(state)
        if DEBUGGER:
            logger.debug('State: %s --> action: %s', pformat(self.qtable[state]), str(action))

        self.stats['actions'].append(action)
        self.qtable_state_action_counter[state][action] += 1
        return action

    def learn(self, sample):
        """ Nothing to do here for learning, only saving the stats. """
        self.stats['rewards'].append(sample['reward'])
        self.qtable_state_action_reward[sample['old_state']][sample['action']].append(
            sample['reward'])
        self.stats['sequence'].append(
            (sample['old_state'], sample['action'], sample['next_state'], sample['reward']))
        if sample['info']:
            sample['info']['from-state'] = sample['old_state']
            self.stats['info'].append(sample['info'])
