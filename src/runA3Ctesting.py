#!/usr/bin/env python3

""" Persuasive Policy Testing environment for RLLIB + SUMO """

import argparse
import cProfile
import io
import json
import logging
import os
import pstats
import sys
import traceback

from pprint import pformat, pprint

import ray

from ray.tune.logger import UnifiedLogger
from utils.logger import DBLogger

from configs.a3c_conf import persuasive_a3c_conf
from learning import persuasivea3c
from environments import persuasivedeepmarlenv

####################################################################################################

def argument_parser():
    """ Argument parser for the trainer"""
    parser = argparse.ArgumentParser(
        description='Reinforcement learning applied to traffic assignment.')
    parser.add_argument(
        '--env', default='MARL', choices=['MARL'],
        help='The MARL environment to use.')
    parser.add_argument(
        '--config', required=True, type=str,
        help='Training configuration.')
    parser.add_argument(
        '--target-checkpoint', required=True,
        help='Path to the target checkpoint to load.')
    parser.add_argument(
        '--dir', required=True,
        help='Path to the directory to use.')
    parser.add_argument(
        '--testing-episodes', default=10, type=int,
        help='Total number of testing episodes.')
    parser.add_argument(
        '--profiler', dest='profiler', action='store_true',
        help='Enables cProfile.')
    parser.set_defaults(profiler=False)
    return parser.parse_args()

ARGS = argument_parser()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

def get_target_checkpoint(checkpoint_dir):
    """ Return target checkpoint, or None. """
    logger.debug('Return the target checkpoint, or None.')
    if not os.path.isdir(checkpoint_dir):
        return None
    for filename in os.listdir(checkpoint_dir):
        if '.' in filename:
            continue
        logger.info('Checkpoint: %s', os.path.join(checkpoint_dir, filename))
        return os.path.join(checkpoint_dir, filename)
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
    """ Testing loop """
    # Args
    print(ARGS)

    # Results
    metrics_dir, checkpoint_dir, debug_dir = results_handler(ARGS)

    # Persuasive A3C Algorithm.
    policy_class = persuasivea3c.PersuasiveA3CTFPolicy
    policy_conf = persuasive_a3c_conf(1, debug_dir) # save every episode

    # REMOVE THE EXPLORATION
    policy_conf['explore'] = False
    # policy_conf['exploration_config']['type'] = 'EpsilonGreedy'

    # Load default Scenario configuration
    experiment_config = load_json_file(ARGS.config)

    # Initialize the simulation.
    # ray.init(memory=52428800, object_store_memory=78643200) ## minimum values
    ray.init()

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
        ray.tune.registry.register_env('marl_env', persuasivedeepmarlenv.env_creator)
        marl_env = persuasivedeepmarlenv.PersuasiveDeepMARLEnv(env_config)
        # marl_env = persuasivedeepmarlenv.PersuasiveDeepMARLEnv.remote(env_config)
    else:
        raise Exception('Unknown environment %s' % ARGS.env)

    # Gen config
    policies = {}
    agent = marl_env.get_agents()[0]
    policies['unique'] = (policy_class,
                          marl_env.get_obs_space(agent),
                          marl_env.get_action_space(agent),
                          {})
    policy_conf['multiagent']['policies'] = policies
    policy_conf['multiagent']['policy_mapping_fn'] = lambda agent_id: 'unique'
    policy_conf['env_config'] = env_config
    pprint(policy_conf)

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

    trainer = persuasivea3c.PersuasiveA3CTrainer(
        env='marl_env', config=policy_conf, logger_creator=default_logger_creator)

    target_checkpoint = get_target_checkpoint(ARGS.target_checkpoint)
    if target_checkpoint is not None:
        logger.info('[Trainer:main] Restoring checkpoint: %s', target_checkpoint)
        trainer.restore(target_checkpoint)
    else:
        raise Exception('Checkpoint {} does not exist.'.format(ARGS.target_checkpoint))

    final_result = None
    for _ in range(ARGS.testing_episodes):
        # Do one step.
        result = trainer.train()
        checkpoint = trainer.save(checkpoint_dir)
        logger.info('[Trainer:main] Checkpoint saved in %s', checkpoint)
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
