#!/usr/bin/env python3

""" Persuasive A3C Configuration """

from pprint import pprint

from learning.persuasivea3c import DEFAULT_CONFIG

from ray.rllib.models import ModelCatalog
from learning.persuasivea3c import PersuasiveCallbacks
from learning.persuasivelstm import RNNModel
from learning.persuasiveactiondistribution import PersuasiveActionDistribution
from learning.persuasivestochasticsampling import PersuasiveStochasticSampling

def persuasive_a3c_conf(rollout_size=10,
                        debug_folder=None,
                        alpha=0.0001,
                        gamma=0.99):
    """
        https://github.com/ray-project/ray/blob/releases/0.8.7/rllib/agents/trainer.py#L44
        https://github.com/ray-project/ray/blob/releases/0.8.7/rllib/agents/a3c/a3c.py#L14
        https://github.com/ray-project/ray/blob/releases/0.8.7/rllib/models/catalog.py#L37
    """

    ModelCatalog.register_custom_model('custom_rrn', RNNModel)
    ModelCatalog.register_custom_action_dist(
        "custom_action_distribution", PersuasiveActionDistribution)

    custom_configuration = DEFAULT_CONFIG

    custom_configuration['batch_mode'] = 'complete_episodes'
    custom_configuration['collect_metrics_timeout'] = 86400 # a day
    custom_configuration['framework'] = 'tf'
    # custom_configuration['gamma'] = gamma
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
    # Training batch size, if applicable. Should be >= rollout_fragment_length.
    # Samples batches will be concatenated together to a batch of this size,
    # which is then passed to SGD.
    custom_configuration['train_batch_size'] = rollout_size

    # === Exploration Settings ===
    custom_configuration['explore'] = True
    # custom_configuration['exploration_config']['type'] = 'EpsilonGreedy'
    custom_configuration['exploration_config']['type'] = 'StochasticSampling'
    # custom_configuration['exploration_config']['type'] = PersuasiveStochasticSampling
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

    # See:
    #  https://docs.ray.io/en/releases-0.8.7/rllib-models.html#custom-action-distributions
    # custom_configuration['model']['custom_action_dist'] = 'custom_action_distribution'
    # custom_configuration['model']['custom_action_dist_par'] = {
    #     'probabilities': [0.9, 0.1],
    # }

    # == Persuasive A3C ==
    custom_configuration['callbacks'] = PersuasiveCallbacks
    # custom_configuration['lr'] = alpha
    custom_configuration['min_iter_time_s'] = 5
    # Divide episodes into fragments of this many steps each during rollouts.
    # Sample batches of this size are collected from rollout workers and
    # combined into a larger batch of `train_batch_size` for learning.
    # For example, given rollout_fragment_length=100 and train_batch_size=1000:
    #   1. RLlib collects 10 fragments of 100 steps each from rollout workers.
    #   2. These fragments are concatenated and we perform an epoch of SGD.
    # When using multiple envs per worker, the fragment size is multiplied by
    # `num_envs_per_worker`. This is since we are collecting steps from
    # multiple envs in parallel. For example, if num_envs_per_worker=5, then
    # rollout workers will return experiences in chunks of 5*100 = 500 steps.
    # The dataflow here can vary per algorithm. For example, PPO further
    # divides the train batch into minibatches for multi-epoch SGD.
    custom_configuration['rollout_fragment_length'] = rollout_size
    custom_configuration['use_gae'] = False
    return custom_configuration
