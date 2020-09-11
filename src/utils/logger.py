#!/usr/bin/env python3

""" RLLIB logger implementation. """

import json
import logging
import os

from ray.tune.logger import Logger, _SafeFallbackEncoder
from ray.tune.result import TRAINING_ITERATION

# %(pathname)s Full pathname of the source file where the logging call was issued(if available).
# %(filename)s Filename portion of pathname.
# %(module)s Module (name portion of filename).
# %(funcName)s Name of function containing the logging call.
# %(lineno)d Source line number where the logging call was issued (if available)
def set_logging(name):
    # 2020-09-11 16:52:15,064	INFO trainer.py:605 -- Tip: s
    logging.basicConfig(
        level=logging.INFO,
        format='(PID=%(process)d)[%(asctime)s][%(levelname)s][%(module)s:L%(lineno)d] %(message)s')
    new_logger = logging.getLogger(name)
    # file handler
    fh = logging.FileHandler('{}.log'.format(name))
    # fh = logging.FileHandler('{}.{}.log'.format(name, os.getpid()))
    fh.setFormatter(
        logging.Formatter(
            '(PID=%(process)d)[%(asctime)s][%(levelname)s][%(module)s:L%(lineno)d] %(message)s'))
    fh.setLevel(logging.DEBUG)
    new_logger.addHandler(fh)
    return new_logger

logger = set_logging(__name__)

class DBLogger(Logger):
    """
    Logging interface for ray.tune. ==> Custom logger for DB

    See:
    https://github.com/ray-project/ray/blob/releases/0.8.4/python/ray/tune/logger.py#L24
    https://github.com/ray-project/ray/blob/releases/0.8.4/python/ray/tune/logger.py#L100

    Arguments:
        config: Configuration passed to all logger creators.
        logdir: Directory for all logger creators to log to.
        trial (Trial): Trial object for the logger to access.
    """

    def on_result(self, result):
        """Given a result, process it and creates the directory tree."""
        # top level dir for this training iteration:
        training = 'training_{}'.format(result[TRAINING_ITERATION])
        self.current_training_dir = os.path.join(self.logdir, 'results', training)
        os.makedirs(self.current_training_dir, exist_ok=True)

        # save config
        config_file = os.path.join(self.current_training_dir, 'config.json')
        with open(config_file, 'w') as fstream:
            json.dump(result['config'], fstream,
                      sort_keys=True, indent=2, cls=_SafeFallbackEncoder)

        # process results
        aggregated_keys = ['episode_reward_mean', 'rewards', 'episode_gtt_mean',
                           'episode_gtt_max', 'episode_gtt_min', 'episodes_this_iter',
                           'episode_elapsed_time_mean', 'timesteps_this_iter',
                           'sumo_steps_this_iter', 'environment_steps_this_iter',]
        misc_keys = ['done', 'timesteps_total', 'episodes_total', 'training_iteration',
                     'experiment_id', 'date', 'timestamp', 'time_this_iter_s',
                     'time_total_s', 'pid', 'hostname', 'node_ip', 'time_since_restore',
                     'timesteps_since_restore', 'iterations_since_restore', 'perf']
        aggregated_values = {}
        misc_values = {}
        for key, val in result.items():
            if key in aggregated_keys:
                aggregated_values[key] = val
            elif key in misc_keys:
                misc_values[key] = val
            else:
                logger.debug('Ignoring key %s', key)

        # save aggregated values
        aggregated_file = os.path.join(self.current_training_dir, 'aggregated-values.json')
        with open(aggregated_file, 'w') as fstream:
            json.dump(aggregated_values, fstream,
                      sort_keys=True, indent=2, cls=_SafeFallbackEncoder)
        # save miscellaneous values
        misc_file = os.path.join(self.current_training_dir, 'misc.json')
        with open(misc_file, 'w') as fstream:
            json.dump(misc_values, fstream,
                      sort_keys=True, indent=2, cls=_SafeFallbackEncoder)

        if 'policies' in result:
            self.process_policies(result['policies'])

    def process_policies(self, policies):
        """ Process and save the data stored in the policies."""
        for agent, policy in policies.items():
            # training level by agent
            training_agent_dir = os.path.join(self.current_training_dir, agent)
            os.makedirs(training_agent_dir, exist_ok=True)

            with open(os.path.join(training_agent_dir, 'best-action.json'), 'w') as fstream:
                json.dump(policy['best-action'], fstream,
                          sort_keys=True, indent=2, cls=_SafeFallbackEncoder)

            with open(os.path.join(training_agent_dir, 'max-qvalue.json'), 'w') as fstream:
                json.dump(policy['max-qvalue'], fstream,
                          sort_keys=True, indent=2, cls=_SafeFallbackEncoder)

            with open(
                    os.path.join(training_agent_dir, 'state-action-counter.json'), 'w') as fstream:
                json.dump(policy['state-action-counter'], fstream,
                          sort_keys=True, indent=2, cls=_SafeFallbackEncoder)

            with open(os.path.join(training_agent_dir, 'qtable.json'), 'w') as fstream:
                json.dump(policy['qtable'], fstream,
                          sort_keys=True, indent=2, cls=_SafeFallbackEncoder)

            # saving cycling vars
            info = policy['stats'].pop('info', None)
            sequence = policy['stats'].pop('sequence', None)

            with open(os.path.join(training_agent_dir, 'stats.json'), 'w') as fstream:
                json.dump(policy['stats'], fstream,
                          sort_keys=True, indent=2, cls=_SafeFallbackEncoder)

            # episode level by agent
            for seq, episode in enumerate(info):
                episode_agent_dir = os.path.join(
                    self.current_training_dir,
                    'episode_{}'.format(seq),
                    agent)
                os.makedirs(episode_agent_dir, exist_ok=True)
                for val in episode:
                    with open(os.path.join(episode_agent_dir, 'info.json'), 'w') as fstream:
                        json.dump(val, fstream,
                                  sort_keys=True, indent=2, cls=_SafeFallbackEncoder)

            for seq, episode in enumerate(sequence):
                episode_agent_dir = os.path.join(
                    self.current_training_dir,
                    'episode_{}'.format(seq),
                    agent)
                os.makedirs(episode_agent_dir, exist_ok=True)
                with open(os.path.join(
                        episode_agent_dir, 'learning-sequence.json'), 'w') as fstream:
                    json.dump(episode, fstream,
                              sort_keys=True, indent=2, cls=_SafeFallbackEncoder)
