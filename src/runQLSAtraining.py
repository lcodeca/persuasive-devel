#!/usr/bin/env python3

""" Persuasive Trainer for RLLIB + SUMO """

import argparse
from copy import deepcopy
import cProfile
import io
import json
import logging
import os
import pstats
import random
import sys
import traceback

from pprint import pformat

import ray

from ray.tune.logger import UnifiedLogger
from utils.logger import DBLogger, set_logging

from configs import qlearning_conf

from environments.rl import marlenvironment, marlenvironmentagentscoop, marlenvironmentlatereward

import learning.ql.qlearningstandalonetrainer as QLStandAlone
import learning.ql.qlearningeligibilitytraces as QLETStandAlone
import learning.ql.pdegreedyqlearningwithet as PDEGQLETStandAlone

####################################################################################################

logger = set_logging(__name__)

####################################################################################################

def argument_parser():
    """ Argument parser for the trainer"""
    parser = argparse.ArgumentParser(
        description='Reinforcement learning applied to traffic assignment.')
    parser.add_argument(
        '--algo', default='QLSA', choices=['QLSA', 'QLET', 'PDEGQLET'],
        help='The RL optimization algorithm to use.')
    parser.add_argument(
        '--env', default='MARL', choices=['MARL', 'MARLCoop', 'LateMARL'],
        help='The MARL environment to use.')
    parser.add_argument(
        '--config', required=True, type=str,
        help='Training configuration.')
    parser.add_argument(
        '--dir', required=True,
        help='Path to the directory to use.')
    parser.add_argument(
        '--checkout-steps', default=10, type=int,
        help='Number of steps between checkouts.')
    parser.add_argument(
        '--training-steps', default=1000, type=int,
        help='Total number of desired training steps.')
    parser.add_argument(
        '--gamma', default=0.9, type=float,
        help="Discount rate, default value is 0.9")
    parser.add_argument(
        '--alpha', default=0.1, type=float,
        help="Learning rate, default value is 0.1")
    parser.add_argument(
        '--epsilon', default=0.01, type=float,
        help="Epsilon, default value is 0.01")
    parser.add_argument(
        '--decay', default=0.9, type=float,
        help="Decay, default value is 0.9")
    parser.add_argument(
        '--action-distr', type=float, nargs='+',
        help="Probability distribution for the epsilon action. Required with PDEGQLET.")
    parser.add_argument(
        '--profiler', dest='profiler', action='store_true',
        help='Enables cProfile.')
    parser.set_defaults(profiler=False)
    return parser.parse_args()

ARGS = argument_parser()
logger = set_logging(runQLSAtraining)

####################################################################################################

def load_json_file(json_file):
    """ Loads a JSON file. """
    logger.debug('Loading %s.', json_file)
    return json.load(open(json_file))

def results_handler(options):
    """ Generate (or retrieve) the results folder for the experiment. """
    logger.debug('Generate (or retrieve) the results folder for the experiment.')
    output_dir = os.path.normpath(options.dir)
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    metrics_dir = os.path.join(output_dir, 'metrics')
    debug_dir = os.path.join(output_dir, 'debug')
    if not os.path.exists(output_dir):
        os.makedirs(metrics_dir)
        os.makedirs(checkpoint_dir)
        os.makedirs(debug_dir)
    metrics_dir = os.path.abspath(metrics_dir)
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    debug_dir = os.path.abspath(debug_dir)
    return metrics_dir, checkpoint_dir, debug_dir

def get_last_checkpoint(checkpoint_dir):
    """ Return the newest available checkpoint, or None. """
    logger.debug('Return the newest available checkpoint, or None.')
    if not os.path.isdir(checkpoint_dir):
        return None
    checkpoints = [os.path.join(checkpoint_dir, folder) for folder in os.listdir(checkpoint_dir)]
    if checkpoints:
        last_checkpoint_dir = max(checkpoints, key=os.path.getmtime)
        for filename in os.listdir(last_checkpoint_dir):
            if '.' in filename:
                continue
            logger.info('Checkpoint: %s',
                        os.path.join(last_checkpoint_dir, filename))
            return os.path.join(last_checkpoint_dir, filename)
    return None

####################################################################################################

SELECTION = [
    'episode_reward_mean', 'episodes_this_iter', 'timesteps_this_iter',
    'sumo_steps_this_iter', 'environment_steps_this_iter', 'rewards',
    'episode_gtt_mean', 'episode_gtt_max', 'episode_gtt_min', 'timesteps_total',
    'episodes_total', 'training_iteration', 'experiment_id', 'date', 'timestamp',
    'time_this_iter_s', 'time_total_s', 'episode_elapsed_time_mean',
]

def print_selected_results(dictionary, keys):
    for key, value in dictionary.items():
        if key in keys:
            logger.info(' %s: %s', key, pformat(value, depth=None, compact=True))

def print_policy_by_agent(policies):
    for agent, policy in policies.items():
        logger.debug('[policies] %s: \n%s', agent, pformat(policy.keys(), depth=None, compact=True))
        keys = ['episodes', 'state', 'qtable', 'max-qvalue', 'best-action',
                'state-action-counter', 'state-action-reward-mean']
        for key, value in policy.items():
            if key in keys:
                logger.info('-> %s: \n%s', key, pformat(value, depth=None, compact=True))
        if 'stats' in policy and 'sequence' in policy['stats']:
            logger.info('Sequence of state-action-state in this checkout.')
            print_sas_sequence(policy['stats']['sequence'])

def print_sas_sequence(sequence):
    for seq, episode in enumerate(sequence):
        logger.info('Sequence of state-action-state from episode %d.', seq)
        before = None
        for state0, action, state1, reward in episode:
            if state0 == before:
                logger.info('A(%s) --> S1(%s) R[%s]', action, state1, reward)
            else:
                logger.info('S0(%s) --> A(%s) --> S1(%s) R[%s]', state0, action, state1, reward)
            before = state1

####################################################################################################

def _main():
    """ Training loop """
    # Args
    logger.info('Arguments: %s', str(ARGS))

    # Results
    metrics_dir, checkpoint_dir, debug_dir = results_handler(ARGS)

    # Algorithm.
    policy_class = None
    policy_conf = None
    policy_params = None
    if ARGS.algo == 'QLSA':
        policy_class = QLStandAlone.EGreedyQLearningPolicy
        policy_conf = qlearning_conf.qlearning_conf(ARGS.checkout_steps, debug_dir)
        policy_params = {
            # Q-Learning defaults
            'alpha': ARGS.alpha,
            'gamma': ARGS.gamma,
            # Epsilon Greedy default
            'epsilon': ARGS.epsilon,
        }
    elif ARGS.algo == 'QLET':
        policy_class = QLETStandAlone.EGreedyQLearningEligibilityTracesPolicy
        policy_conf = qlearning_conf.qlearning_conf(ARGS.checkout_steps, debug_dir)
        policy_params = {
            # Q-Learning defaults
            'alpha': ARGS.alpha,
            'gamma': ARGS.gamma,
            # Eligibility traces defaults
            'decay': ARGS.decay,
            # Epsilon Greedy default
            'epsilon': ARGS.epsilon,
        }
    elif ARGS.algo == 'PDEGQLET':
        policy_class = PDEGQLETStandAlone.PDEGreedyQLearningETPolicy
        policy_conf = qlearning_conf.qlearning_conf(ARGS.checkout_steps, debug_dir)
        policy_params = {
            # Q-Learning defaults
            'alpha': ARGS.alpha,
            'gamma': ARGS.gamma,
            # Eligibility traces defaults
            'decay': ARGS.decay,
            # Epsilon Greedy default
            'epsilon': ARGS.epsilon,
            # Probability distribution for the epsilon-action
            'actions-distribution': ARGS.action_distr,
        }
    else:
        raise Exception('Unknown algorithm %s' % ARGS.algo)

    # Load default Scenario configuration
    experiment_config = load_json_file(ARGS.config)

    # Initialize the simulation.
    ray.init(memory=52428800, object_store_memory=78643200) ## minimum values

    # Associate the agents with something
    agent_init = load_json_file(experiment_config['agents_init_file'])
    env_config = {
        'metrics_dir': metrics_dir,
        'checkpoint_dir': checkpoint_dir,
        'agent_init': agent_init,
        'scenario_config': experiment_config['marl_env_config'],
    }
    marl_env = None
    if ARGS.env == 'MARL':
        ray.tune.registry.register_env('marl_env', marlenvironment.env_creator)
        marl_env = marlenvironment.PersuasiveMultiAgentEnv(env_config)
    elif ARGS.env == 'MARLCoop':
        ray.tune.registry.register_env('marl_env', marlenvironmentagentscoop.env_creator)
        marl_env = marlenvironmentagentscoop.AgentsCoopMultiAgentEnv(env_config)
    elif ARGS.env == 'LateMARL':
        ray.tune.registry.register_env('marl_env', marlenvironmentlatereward.env_creator)
        marl_env = marlenvironmentlatereward.LateRewardMultiAgentEnv(env_config)
    else:
        raise Exception('Unknown environment %s' % ARGS.env)

    # Gen config
    policies = {}
    for agent in marl_env.get_agents():
        agent_policy_params = deepcopy(policy_params)
        from_val, to_val = agent_init[agent]['init']
        agent_policy_params['init'] = lambda: random.randint(from_val, to_val)
        agent_policy_params['actions'] = marl_env.get_set_of_actions(agent)
        agent_policy_params['seed'] = agent_init[agent]['seed']
        policies[agent] = (policy_class,
                           marl_env.get_obs_space(agent),
                           marl_env.get_action_space(agent),
                           agent_policy_params)
    policy_conf['multiagent'] = {
        'policies': policies,
        'policy_mapping_fn': lambda agent_id: agent_id,
    }
    policy_conf['env_config'] = env_config
    logger.info('Configuration: \n%s', pformat(policy_conf))

    def default_logger_creator(config):
        """
            Creates a Unified logger with a default logdir prefix
            containing the agent name and the env id
        """
        log_dir = os.path.join(os.path.normpath(ARGS.dir), 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return UnifiedLogger(config, log_dir) # loggers = None) >> Default loggers

    def dblogger_logger_creator(config):
        """
            Creates a Unified logger with a default logdir prefix
            containing the agent name and the env id
        """
        log_dir = os.path.join(os.path.normpath(ARGS.dir), 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return UnifiedLogger(config, log_dir, loggers=[DBLogger])

    trainer = None
    if ARGS.algo == 'QLSA':
        trainer = QLStandAlone.QLearningTrainer(
            env='marl_env', config=policy_conf, logger_creator=dblogger_logger_creator)
    elif ARGS.algo == 'QLET':
        trainer = QLETStandAlone.QLearningEligibilityTracesTrainer(
            env='marl_env', config=policy_conf, logger_creator=dblogger_logger_creator)
    elif ARGS.algo == 'PDEGQLET':
        trainer = PDEGQLETStandAlone.PDEGreedyQLearningETTrainer(
            env='marl_env', config=policy_conf, logger_creator=dblogger_logger_creator)
    else:
        raise Exception('Unknown algorithm %s' % ARGS.algo)

    last_checkpoint = get_last_checkpoint(checkpoint_dir)
    if last_checkpoint is not None:
        logger.info('[Trainer:main] Restoring checkpoint: %s', last_checkpoint)
        trainer.restore(last_checkpoint)

    steps = 0
    final_result = None

    while steps < ARGS.training_steps:
        # Do one step.
        result = trainer.train()
        checkpoint = trainer.save(checkpoint_dir)
        logger.info('[Trainer:main] Checkpoint saved in %s', checkpoint)
        # steps += result['info']['num_steps_trained']
        steps += result['timesteps_this_iter'] # is related to 'timesteps_total' that is the same
                                               # as result['info']['num_steps_sampled']
        final_result = result

    print_selected_results(final_result, SELECTION)
    # print_policy_by_agent(final_result['policies'])

if __name__ == '__main__':
    ret = 0
    ## ========================              PROFILER              ======================== ##
    if ARGS.profiler:
        profiler = cProfile.Profile()
        profiler.enable()
    ## ========================              PROFILER              ======================== ##
    try:
        _main()
    except Exception: # traci.exceptions.TraCIException:
        ret = 666
        EXC_TYPE, EXC_VALUE, EXC_TRACEBACK = sys.exc_info()
        traceback.print_exception(EXC_TYPE, EXC_VALUE, EXC_TRACEBACK, file=sys.stdout)
    finally:
        ray.shutdown()
        ## ========================          PROFILER              ======================== ##
        if ARGS.profiler:
            profiler.disable()
            results = io.StringIO()
            pstats.Stats(profiler, stream=results).sort_stats('cumulative').print_stats(50)
            logger.info('Profiler: \n%s', results.getvalue())
        ## ========================          PROFILER              ======================== ##
        sys.exit(ret)
