#!/usr/bin/env python3

""" Persuasive A3C Configuration """

from pprint import pprint

import ray
from ray.rllib.evaluation.metrics import collect_episodes, collect_metrics, summarize_episodes
from ray.rllib.models import ModelCatalog

from learning.a3c.persuasivea3c import DEFAULT_CONFIG, PersuasiveCallbacks
from learning.a3c.persuasivelstm import RNNModel
from learning.a3c.persuasiveactiondistribution import PersuasiveActionDistribution

def custom_eval_function(trainer, eval_workers):
    """Example of a custom evaluation function.
    Arguments:
        trainer (Trainer): trainer class to evaluate.
        eval_workers (WorkerSet): evaluation workers.
    Returns:
        metrics (dict): evaluation metrics dict.
    """

    print('\n\n\n\n custom_eval_function \n\n\n\n')
    # We configured 1 local eval workers in the training config.
    worker = eval_workers.local_worker()
    # worker = eval_workers.remote_workers()
    # Reset the worker environment.
    worker.foreach_env(lambda env: env.reset())

    for i in range(2):
        print('\n\n\n\n Custom evaluation round {} \n\n\n\n'.format(i))
        # Calling .sample() runs exactly one episode per worker due to how the
        # eval workers are configured.
        ray.get([w.sample() for w in eval_workers.local_workers()])

    # Collect the accumulated episodes on the workers, and then summarize the
    # episode stats into a metrics dict.
    # episodes, _ = collect_episodes(
    #     remote_workers=eval_workers.remote_workers(), timeout_seconds=99999)
    # You can compute metrics from the episodes manually, or use the
    # convenient `summarize_episodes()` utility:
    # metrics = summarize_episodes(episodes)
    # Note that the above two statements are the equivalent of:
    metrics = collect_metrics(eval_workers.local_worker(),
                              eval_workers.remote_workers())

    # You can also put custom values in the metrics dict.
    print('\n\n\n\n')
    pprint(metrics)
    print('\n\n\n\n')
    return metrics

def persuasive_a3c_conf(rollout_size=10,
                        agents=100,
                        debug_folder=None,
                        eval_folder=None,
                        alpha=0.0001,
                        gamma=0.99):
    """
        https://github.com/ray-project/ray/blob/releases/1.0.0/rllib/agents/trainer.py#L44
        https://github.com/ray-project/ray/blob/releases/1.0.0/rllib/agents/a3c/a3c.py#L14
        https://github.com/ray-project/ray/blob/releases/1.0.0/rllib/models/catalog.py#L37
    """

    ModelCatalog.register_custom_model('custom_rrn', RNNModel)
    ModelCatalog.register_custom_action_dist(
        "custom_action_distribution", PersuasiveActionDistribution)

    custom_configuration = DEFAULT_CONFIG

    custom_configuration['collect_metrics_timeout'] = 86400 # a day
    custom_configuration['framework'] = 'tf'
    custom_configuration['ignore_worker_failures'] = True
    custom_configuration['log_level'] = 'WARN'
    custom_configuration['monitor'] = True
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

    # === Environment Settings ===
    custom_configuration['batch_mode'] = 'complete_episodes'
    custom_configuration['gamma'] = gamma
    custom_configuration['lr'] = alpha
    custom_configuration['no_done_at_end'] = False
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
    # Training batch size, if applicable. Should be >= rollout_fragment_length.
    # Samples batches will be concatenated together to a batch of this size,
    # which is then passed to SGD.
    custom_configuration['train_batch_size'] = rollout_size * agents

    # === Exploration Settings ===
    # https://github.com/ray-project/ray/blob/releases/1.0.0/rllib/utils/exploration/stochastic_sampling.py
    # custom_configuration['exploration_config']['type'] = 'StochasticSampling'

    # https://github.com/ray-project/ray/blob/releases/1.0.0/rllib/utils/exploration/epsilon_greedy.py
    custom_configuration['exploration_config']['type'] = 'EpsilonGreedy'
    custom_configuration['exploration_config']['initial_epsilon'] = 1.0
    custom_configuration['exploration_config']['final_epsilon'] = 0.0001

    # ==================== MODEL - DEFAULT ====================
    # custom_configuration['model']['fcnet_hiddens'] = [64, 64]
    # === Built-in options ===
    # Filter config. List of [out_channels, kernel, stride] for each filter
    # custom_configuration['model']['conv_filters'] = None
    # Nonlinearity for built-in convnet
    # custom_configuration['model']['conv_activation'] = "relu"
    # Nonlinearity for fully connected net (tanh, relu)
    # custom_configuration['model']['fcnet_activation'] = "tanh"
    # Number of hidden layers for fully connected net
    # custom_configuration['model']['fcnet_hiddens'] = [64, 64]
    # For DiagGaussian action distributions, make the second half of the model
    # outputs floating bias variables instead of state-dependent. This only
    # has an effect is using the default fully connected net.
    # custom_configuration['model']['free_log_std'] = False
    # Whether to skip the final linear layer used to resize the hidden layer
    # outputs to size `num_outputs`. If True, then the last hidden layer
    # should already match num_outputs.
    # custom_configuration['model']['no_final_linear'] = False
    # Whether layers should be shared for the value function.
    # custom_configuration['model']['vf_share_layers'] = True

    # == LSTM ==
    # Whether to wrap the model with an LSTM.
    # custom_configuration['model']['use_lstm'] = False
    # Max seq len for training the LSTM, defaults to 20.
    # custom_configuration['model']['max_seq_len'] = 20
    # Size of the LSTM cell.
    # custom_configuration['model']['lstm_cell_size'] = 64
    # Whether to feed a_{t-1}, r_{t-1} to LSTM.
    # custom_configuration['model']['lstm_use_prev_action_reward'] = False
    # When using modelv1 models with a modelv2 algorithm, you may have to
    # define the state shape here (e.g., [256, 256]).
    # custom_configuration['model']['state_shape'] = None # [64, 64]

    # == Atari ==
    # Whether to enable framestack for Atari envs
    # custom_configuration['model']['framestack'] = True
    # Final resized frame dimension
    # custom_configuration['model']['dim'] = 84
    # (deprecated) Converts ATARI frame to 1 Channel Grayscale image
    # custom_configuration['model']['grayscale'] = False
    # (deprecated) Changes frame to range from [-1, 1] if true
    # custom_configuration['model']['zero_mean'] = True

    # === Options for custom models ===
    # Name of a custom model to use
    custom_configuration['model']['custom_model'] = 'custom_rrn'
    # Extra options to pass to the custom classes.
    # These will be available in the Model's
    custom_configuration['model']['custom_model_config'] = {}
    # Name of a custom action distribution to use.
    # See: https://docs.ray.io/en/releases-1.0.0/rllib-models.html#custom-action-distributions
    custom_configuration['model']['custom_action_dist'] = 'custom_action_distribution'

    # == OPTIMIZER ==
    # Arguments to pass to the policy optimizer. These vary by optimizer.
    # custom_configuration['optimizer'] = {}

    # == Persuasive A3C ==
    custom_configuration['callbacks'] = PersuasiveCallbacks
    custom_configuration['min_iter_time_s'] = 5

    custom_configuration['use_gae'] = True

    # === Evaluation Settings ===
    # Evaluate with every `evaluation_interval` training iterations.
    # The evaluation stats will be reported under the "evaluation" metric key.
    # Note that evaluation is currently not parallelized, and that for Ape-X
    # metrics are already only reported for the lowest epsilon workers.
    custom_configuration['evaluation_interval'] = 5

    # Number of episodes to run per evaluation period. If using multiple
    # evaluation workers, we will run at least this many episodes total.
    custom_configuration['evaluation_num_episodes'] = 5

    # Internal flag that is set to True for evaluation workers.
    # DEFAUTL: 'in_evaluation': False,

    # Typical usage is to pass extra args to evaluation env creator
    # and to disable exploration by computing deterministic actions.
    # IMPORTANT NOTE: Policy gradient algorithms are able to find the optimal
    # policy, even if this is a stochastic one. Setting 'explore=False' here
    # will result in the evaluation workers not using this optimal policy!
    custom_configuration['evaluation_config']['explore'] = False
    custom_configuration['evaluation_config']['lr'] = 0
    custom_configuration['evaluation_config']['num_gpus_per_worker'] = 0
    custom_configuration['evaluation_config']['num_gpus'] = 0
    custom_configuration['evaluation_config']['output'] = eval_folder
    # custom_configuration['evaluation_config']['env_config'] = {...},

    # Number of parallel workers to use for evaluation. Note that this is set
    # to zero by default, which means evaluation will be run in the trainer
    # process. If you increase this, it will increase the Ray resource usage
    # of the trainer since evaluation workers are created separately from
    # rollout workers.
    custom_configuration['evaluation_num_workers'] = 1

    # Customize the evaluation method. This must be a function of signature
    # (trainer: Trainer, eval_workers: WorkerSet) -> metrics: dict. See the
    # Trainer._evaluate() method to see the default implementation. The
    # trainer guarantees all eval workers have the latest policy state before
    # this function is called.
    custom_configuration['custom_eval_function'] = None #custom_eval_function

    return custom_configuration
