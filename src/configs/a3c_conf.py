#!/usr/bin/env python3

""" Persuasive A3C Configuration """

from pprint import pprint

from learning.persuasivea3c import DEFAULT_CONFIG

from ray.rllib.models import ModelCatalog
from learning.lstm import RNNModel

def persuasive_a3c_conf(tr_steps=1, debug_folder=None, alpha=0.0001, gamma=0.99):
    """
        https://github.com/ray-project/ray/blob/releases/0.8.7/rllib/agents/trainer.py#L44
        https://github.com/ray-project/ray/blob/releases/0.8.7/rllib/agents/a3c/a3c.py#L14
        https://github.com/ray-project/ray/blob/releases/0.8.7/rllib/models/catalog.py#L37
    """

    ModelCatalog.register_custom_model('custom_rrn', RNNModel)

    custom_configuration = DEFAULT_CONFIG

    custom_configuration['batch_mode'] = 'complete_episodes'
    custom_configuration['collect_metrics_timeout'] = 86400 # a day
    custom_configuration['gamma'] = gamma
    custom_configuration['ignore_worker_failures'] = True
    custom_configuration['log_level'] = 'WARN'
    custom_configuration['monitor'] = True
    custom_configuration['no_done_at_end'] = False
    custom_configuration['num_cpus_for_driver'] = 1
    custom_configuration['num_cpus_per_worker'] = 1
    custom_configuration['num_envs_per_worker'] = 1
    custom_configuration['num_gpus_per_worker'] = 1
    custom_configuration['num_gpus'] = 1
    custom_configuration['num_workers'] = 1
    custom_configuration['output'] = debug_folder
    custom_configuration['remote_env_batch_wait_ms'] = 1000
    custom_configuration['remote_worker_envs'] = False
    custom_configuration['seed'] = 42
    custom_configuration['timesteps_per_iteration'] = 1
    custom_configuration['train_batch_size'] = tr_steps

    # === Exploration Settings ===
    # Default exploration behavior, iff `explore`=None is passed into
    # compute_action(s).
    # Set to False for no exploration behavior (e.g., for evaluation).
    custom_configuration['explore'] = True
    # Provide a dict specifying the Exploration object's config.
    # The Exploration class to use. In the simplest case, this is the name
    # (str) of any class present in the `rllib.utils.exploration` package.
    # You can also provide the python class directly or the full location
    # of your class (e.g. "ray.rllib.utils.exploration.epsilon_greedy.
    # EpsilonGreedy").
    custom_configuration['exploration_config']['type'] = 'StochasticSampling'
    # Add constructor kwargs here (if any).

    # == MODEL - DEFAULT ==
    # custom_configuration['model']['fcnet_hiddens'] = [64, 64]

    # == MODEL - LSTM ==
    # custom_configuration['model']['lstm_cell_size'] = 64
    # custom_configuration['model']['max_seq_len'] = 2
    # custom_configuration['model']['state_shape'] = [64, 64]
    # custom_configuration['model']['use_lstm'] = True

    # == MODEL - CUSTOM ==
    custom_configuration['model']['custom_model'] = 'custom_rrn'

    # == Persuasive A3C ==
    custom_configuration['lr'] = alpha
    custom_configuration['min_iter_time_s'] = 5
    custom_configuration['rollout_fragment_length'] = tr_steps
    custom_configuration['use_gae'] = False

    pprint(custom_configuration)
    return custom_configuration
