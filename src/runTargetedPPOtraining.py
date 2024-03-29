#!/usr/bin/env python3

"""
Persuasive Trainer for RLLIB + SUMO

See:
 - https://docs.ray.io/en/latest/rllib-algorithms.html#proximal-policy-optimization-ppo
 - https://github.com/ray-project/ray/blob/releases/1.0.0/rllib/agents/ppo/ppo.py
 - https://github.com/ray-project/ray/blob/releases/1.0.0/rllib/agents/ppo/ppo_tf_policy.py
"""

import os
# os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["RAY_DEBUG_DISABLE_MEMORY_MONITOR"] = str(1)

import argparse
from copy import deepcopy
import cProfile
import io
import json
import logging
import pstats
import shutil
import sys
import traceback

from pprint import pformat, pprint

import numpy as np
from numpyencoder import NumpyEncoder

import ray

from ray.tune.logger import NoopLogger, UnifiedLogger
from utils.logger import set_logging

import ray.rllib.agents.ppo as ppo
from configs.ppo_conf import persuasive_ppo_conf
from environments.stochasticdeeprl import stochasticdeepmarlenv

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
        '--dir', required=True,
        help='Path to the directory to use.')
    parser.add_argument(
        '--rollout-size', default=1, type=int,
        help='Size of the rollout fragment batch.')
    parser.add_argument(
        '--training-iterations', default=1000, type=int,
        help='Total number of desired training iterations.')
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
    ######## RAY
    parser.add_argument(
        '--ray-mem-gb', default=10, type=int,
        help='RAY Init _memory parameter in GB.')
    parser.add_argument(
        '--ray-store-gb', default=10, type=int,
        help='RAY Init object_store_memory parameter in GB')
    parser.add_argument(
        '--ray-cpus', default=str(os.cpu_count()), type=int,
        help='RAY Init num_cpus parameter.')
    parser.add_argument(
        '--ray-gpus', default=0, type=int,
        help='RAY Init num_gpus parameter.')
    ######## Profiling
    parser.add_argument(
        '--profiler', dest='profiler', action='store_true',
        help='Enables cProfile.')
    parser.set_defaults(profiler=False)
    return parser.parse_args()

ARGS = argument_parser()
logger = set_logging('runPPOtraining')

####################################################################################################

class NPEncoder(NumpyEncoder):
    def default(self, obj):
        try:
            encoded_value = super(NPEncoder, self).default(obj)
            return encoded_value
        except:
            logger.debug('%s ---> %s', str(type(obj)), str(obj))
            return str(obj)

def load_json_file(json_file):
    """ Loads a JSON file. """
    logger.debug('Loading %s.', json_file)
    return json.load(open(json_file))

####################################################################################################

CHECKPOINT_METRICS = [
    'max_policy_reward_min', 'max_policy_reward_mean', # this is a unique value
    'max_arrival_mean', 'max_arrival_min',
    'min_wait_mean', 'min_wait_min',
    'max_on_time_agent_mean', 'max_on_time_agent_min',
    'min_total_loss', 'min_policy_loss', 'min_policy_entropy',
]

STOPPING_METRICS = [
    'max_policy_reward_min', 'max_policy_reward_mean', # this is a unique value
    'max_arrival_mean', 'max_arrival_min',
    'min_wait_mean', 'min_wait_min',
    'max_on_time_agent_mean', 'max_on_time_agent_min',
    'min_total_loss', 'min_policy_loss', 'min_policy_entropy',
]

CURRENT_METRICS = {
    'min_total_loss': {
        'check': lambda new, old: old is None or new < old,
        'get': lambda res: res['info']['learner']['unique']['total_loss'],
        'value': None,
    },
    'min_policy_loss': {
        'check': lambda new, old: old is None or new < old,
        'get': lambda res: res['info']['learner']['unique']['policy_loss'],
        'value': None,
    },
    'min_policy_entropy': {
        'check': lambda new, old: old is None or new < old,
        'get': lambda res: res['info']['learner']['unique']['entropy'],
        'value': None,
    },
    'max_policy_reward_min': {
        'check': lambda new, old: old is None or new > old,
        'get': lambda res: res['evaluation']['policy_reward_min']['unique'],
        'value': None,
    },
    'max_policy_reward_mean': {
        'check': lambda new, old: old is None or new > old,
        'get': lambda res: res['evaluation']['policy_reward_mean']['unique'],
        'value': None,
    },
    'max_arrival_mean': {
        'check': lambda new, old: old is None or new > old,
        'get': lambda res: res['evaluation']['custom_metrics']['episode_average_arrival_mean'],
        'value': None,
    },
    'max_arrival_min': {
        'check': lambda new, old: old is None or new > old,
        'get': lambda res: res['evaluation']['custom_metrics']['episode_average_arrival_min'],
        'value': None,
    },
    'min_wait_mean': {
        'check': lambda new, old: old is None or new < old,
        'get': lambda res: abs(res['evaluation']['custom_metrics']['episode_average_wait_min']),
        'value': None,
    },
    'min_wait_min': {
        'check': lambda new, old: old is None or new < old,
        'get': lambda res: abs(res['evaluation']['custom_metrics']['episode_average_wait_min']),
        'value': None,
    },
    'max_on_time_agent_mean': {
        'check': lambda new, old: old is None or new > old,
        'get': lambda res: abs(res['evaluation']['custom_metrics']['episode_on_time_agents_mean']),
        'value': None,
    },
    'max_on_time_agent_min': {
        'check': lambda new, old: old is None or new > old,
        'get': lambda res: abs(res['evaluation']['custom_metrics']['episode_on_time_agents_min']),
        'value': None,
    },
}

def cleanup(checkpoint_dir):
    """ Remove all the previous checkpoints from the directory. """
    shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir)
    return True

def get_last_best_of(checkpoint_dir):
    """ Return the info from the checkpoint directory, or None. """
    logger.debug('Return the info from the checkpoint directory, or None. ')
    expected_results = os.path.join(checkpoint_dir, 'info.json')
    if not os.path.isfile(expected_results):
        return None
    res = load_json_file(expected_results)
    logger.info('Restored info %s.', pformat(res))
    return float(res['value'])

####################################################################################################

def results_handler(options):
    """ Generate (or retrieve) the results folder for the experiment. """
    logger.debug('Generate (or retrieve) the results folder for the experiment.')
    output_dir = os.path.normpath(options.dir)
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    best_checkpoint_dir = os.path.join(output_dir, 'best_checkpoints')
    metrics_dir = os.path.join(output_dir, 'metrics')
    debug_dir = os.path.join(output_dir, 'debug')
    eval_dir = os.path.join(output_dir, 'eval')
    if not os.path.exists(output_dir):
        os.makedirs(metrics_dir)
        os.makedirs(checkpoint_dir)
        for metric in CHECKPOINT_METRICS:
            os.makedirs(os.path.join(best_checkpoint_dir, metric))
        os.makedirs(debug_dir)
        os.makedirs(eval_dir)
    metrics_dir = os.path.abspath(metrics_dir)
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    best_checkpoint_dir = os.path.abspath(best_checkpoint_dir)
    debug_dir = os.path.abspath(debug_dir)
    eval_dir = os.path.abspath(eval_dir)
    return metrics_dir, checkpoint_dir, best_checkpoint_dir, debug_dir, eval_dir

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

COMPLETE = [
    'episode_reward_max', 'episode_reward_min', 'episode_reward_mean', 'episode_len_mean',
    'episodes_this_iter', 'policy_reward_min', 'policy_reward_max', 'policy_reward_mean',
    'custom_metrics', 'hist_stats', 'sampler_perf', 'off_policy_estimator',
    'num_healthy_workers', 'timesteps_total', 'timers', 'info', 'done', 'episodes_total',
    'training_iteration', 'experiment_id', 'date', 'timestamp', 'time_this_iter_s',
    'time_total_s', 'pid', 'hostname', 'node_ip', 'config', 'time_since_restore',
    'timesteps_since_restore', 'iterations_since_restore', 'perf',
]

SELECTION = [
    'episode_len_mean', 'episodes_this_iter', 'episodes_total',
    'policy_reward_min', 'policy_reward_max', 'policy_reward_mean', 'info',
    'timesteps_total', 'training_iteration', 'experiment_id', 'date', 'timestamp',
    'time_since_restore', 'timesteps_since_restore', 'iterations_since_restore',
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

GB = 1024 * 1024 * 1024

def _main():
    """ Training loop """
    # Args
    logger.info('Arguments: %s', str(ARGS))

    # Results
    metrics_dir, checkpoint_dir, best_checkpoint_dir, debug_dir, eval_dir = results_handler(ARGS)

    # Initialize the simulation.
    # ray.init()
    # ray.init(memory=52428800, object_store_memory=78643200) ## minimum values
    # ray.init(address='auto', _redis_password='5241590000000000') ## attach
    ray.init(
        num_cpus=ARGS.ray_cpus, num_gpus=ARGS.ray_gpus,
        _memory=ARGS.ray_mem_gb*GB, object_store_memory=ARGS.ray_store_gb*GB)

    # Load default Scenario configuration
    experiment_config = load_json_file(ARGS.config)

    # Associate the agents with something
    env_config = {
        'metrics_dir': metrics_dir,
        'checkpoint_dir': checkpoint_dir,
        'agent_init': load_json_file(experiment_config['agents_init']),
        'scenario_config': experiment_config['marl_env_config'],
    }
    ## fix the config for learning:
    env_config['agent_init']['eval'] = {}

    marl_env = None
    if ARGS.env == 'MARL':
        ray.tune.registry.register_env('marl_env', stochasticdeepmarlenv.env_creator)
        marl_env = stochasticdeepmarlenv.StochasticPersuasiveDeepMARLEnv(env_config)
    else:
        raise Exception('Unknown environment %s' % ARGS.env)

    # Persuasive A3C Algorithm.
    policy_class = ppo.PPOTFPolicy
    policy_conf = persuasive_ppo_conf(
        rollout_size=ARGS.rollout_size, agents=len(marl_env.get_agents()),
        debug_folder=debug_dir, eval_folder=eval_dir,
        alpha=ARGS.alpha, gamma=ARGS.gamma)
    # Gen config
    agent = marl_env.get_agents()[0]
    policies = {
        'unique': (policy_class,
                   marl_env.get_obs_space(agent),
                   marl_env.get_action_space(agent),
                   {})
    }
    policy_conf['multiagent']['policies'] = policies
    policy_conf['multiagent']['policy_mapping_fn'] = lambda agent_id: 'unique'
    policy_conf['env_config'] = env_config
    # policy_conf['evaluation_config']['env_config'] = {
    #     'metrics_dir': metrics_dir,
    #     'checkpoint_dir': checkpoint_dir,
    #     'agent_init': load_json_file(experiment_config['agents_init']),
    #     'scenario_config': experiment_config['marl_env_config'],
    # }
    logger.info('Configuration: \n%s', pformat(policy_conf))

    def default_logger_creator(config):
        """
            Creates a Unified logger with a default logdir prefix
            containing the agent name and the env id
        """
        log_dir = os.path.join(os.path.normpath(ARGS.dir), 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return UnifiedLogger(config, log_dir, loggers=[NoopLogger])

    trainer = ppo.PPOTrainer(
        # env=deepmarlenvironment.PersuasiveDeepMARLEnv,
        env='marl_env', config=policy_conf, logger_creator=default_logger_creator)

    last_checkpoint = get_last_checkpoint(checkpoint_dir)
    if last_checkpoint is not None:
        trainer.restore(last_checkpoint)
        logger.info('Restored checkpoint: %s', last_checkpoint)

    # Restoring the latest best metrics
    for metric in CHECKPOINT_METRICS:
        CURRENT_METRICS[metric]['value'] = get_last_best_of(os.path.join(best_checkpoint_dir,
                                                                         metric))
    logger.info('Restored metrics: \n%s', pformat(CURRENT_METRICS))

    counter = 0
    unchanged_window = 0
    final_result = None
    while counter < ARGS.training_iterations:
        # Do one training step.
        result = trainer.train()
        checkpoint = trainer.save(checkpoint_dir)
        logger.info('Checkpoint saved in %s', checkpoint)
        counter = result['iterations_since_restore']
        # counter = result['training_iteration']
        # steps += result['info']['num_steps_trained']
        # steps += result['timesteps_this_iter']
        final_result = result
        print_selected_results(result, SELECTION)
        metric_file = os.path.join(metrics_dir,
                                   'metrics_{}.json'.format(result['training_iteration']))
        with open(metric_file, 'w') as fstream:
            # the evaluation metrica are not saved in 'results.json'
            json.dump(result, fstream, cls=NPEncoder)
            # fstream.write('\n')
            print('############################# METRIC SAVED #############################')
        ############################################################################################
        if 'evaluation' not in result:
            continue
        changes = False
        for metric in CHECKPOINT_METRICS:
            old = CURRENT_METRICS[metric]['value']
            new = CURRENT_METRICS[metric]['get'](result)
            # if np.isnan(new):
            #     pprint(result['evaluation'])
            #     raise Exception(metric, old, new)
            if CURRENT_METRICS[metric]['check'](new, old):
                # Save the "best" checkout
                if metric in STOPPING_METRICS:
                    changes = True
                CURRENT_METRICS[metric]['value'] = new
                cleanup(os.path.join(best_checkpoint_dir, metric))
                current_checkpoint = trainer.save(os.path.join(best_checkpoint_dir, metric))
                current_info_file = os.path.join(best_checkpoint_dir, metric, 'info.json')
                current_value = {'value': str(new)}
                with open(current_info_file, 'w') as fstream:
                    json.dump(current_value, fstream, indent=4)
                if old is None:
                    old = -1.0
                logger.info('UPDATING %s: %.2f (%.2f). Checkpoint saved in %s',
                            metric, new, old, current_checkpoint)
            else:
                logger.info('UNCHANGED %s ---> Best: %.2f - New: %.2f', metric, old, new)
        if changes:
            unchanged_window = 0
        else:
            unchanged_window += 1
            logger.info(
                'Nothing has changed for the last %d training runs in the monitored metrics [%s].',
                unchanged_window, str(STOPPING_METRICS))
        if unchanged_window >= 10:
            break
        ############################################################################################

    # pprint(final_result)
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
    except Exception: # traci.exceptions.TraCIException: libsumo.libsumo.TraCIException:
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
