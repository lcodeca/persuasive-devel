#!/usr/bin/env python3

""" Persuasive PPO Configuration """

from pprint import pprint

import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.models import ModelCatalog

from learning.a3c.persuasivea3c import PersuasiveCallbacks
from learning.a3c.persuasiveactiondistribution import PersuasiveActionDistribution

def persuasive_ppo_conf(rollout_size=10,
                        agents=100,
                        debug_folder=None,
                        eval_folder=None,
                        alpha=5e-5,
                        gamma=0.99):
    """
        https://github.com/ray-project/ray/blob/releases/1.0.0/rllib/agents/trainer.py#L44
        https://github.com/ray-project/ray/blob/releases/1.0.0/rllib/agents/ppo/ppo.py#L15
        https://github.com/ray-project/ray/blob/releases/1.0.0/rllib/models/catalog.py#L37
    """

    ModelCatalog.register_custom_action_dist(
        'custom_action_distribution', PersuasiveActionDistribution)

    custom_configuration = ppo.DEFAULT_CONFIG.copy()

    # custom_configuration['collect_metrics_timeout'] = 86400 # a day
    custom_configuration['framework'] = 'tf'
    custom_configuration['ignore_worker_failures'] = True
    custom_configuration['log_level'] = 'WARN'
    custom_configuration['monitor'] = True
    custom_configuration['num_cpus_for_driver'] = 1
    custom_configuration['num_cpus_per_worker'] = 1
    custom_configuration['num_envs_per_worker'] = 1
    custom_configuration['output'] = debug_folder
    # custom_configuration['remote_env_batch_wait_ms'] = 1000
    # custom_configuration['remote_worker_envs'] = False
    custom_configuration['seed'] = 42

    # === Parallelism ===
    # Number of workers for collecting samples with. This only makes sense
    # to increase if your environment is particularly slow to sample, or if
    # you"re using the Async or Ape-X optimizers.
    custom_configuration['num_workers'] = 4
    custom_configuration['num_gpus_per_worker'] = 0
    # Prevent iterations from going lower than this time span
    # custom_configuration['min_iter_time_s'] = 1

    # === Environment Settings ===
    custom_configuration['batch_mode'] = 'complete_episodes'
    custom_configuration['callbacks'] = PersuasiveCallbacks
    custom_configuration['gamma'] = gamma
    custom_configuration['lr'] = alpha
    custom_configuration['lr_schedule'] = None
    custom_configuration['no_done_at_end'] = False

    # === Exploration Settings ===
    # custom_configuration['exploration_config'] = {}

    # # https://github.com/ray-project/ray/blob/releases/1.0.0/rllib/utils/exploration/epsilon_greedy.py
    # custom_configuration['exploration_config']['type'] = 'EpsilonGreedy'
    # custom_configuration['exploration_config']['initial_epsilon'] = 1.0
    # custom_configuration['exploration_config']['final_epsilon'] = 0.02
    # custom_configuration['exploration_config']['epsilon_timesteps'] = 10000

    # https://github.com/ray-project/ray/blob/releases/1.0.0/rllib/utils/exploration/soft_q.py
    # custom_configuration['exploration_config']['type'] = 'SoftQ'
    # custom_configuration['exploration_config']['temperature'] = 1.0 # Default

    # Name of a custom action distribution to use.
    # See: https://docs.ray.io/en/releases-1.0.0/rllib-models.html#custom-action-distributions
    # custom_configuration['model']['custom_action_dist'] = 'custom_action_distribution'

    # === PPO Model Settings ===
    # Should use a critic as a baseline (otherwise don't use value baseline;
    # required for using GAE).
    custom_configuration['use_critic'] = True
    # If true, use the Generalized Advantage Estimator (GAE)
    # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
    custom_configuration['use_gae'] = True
    # The GAE(lambda) parameter.
    custom_configuration['lambda'] = 1.0
    # Initial coefficient for KL divergence.
    custom_configuration['kl_coeff'] = 0.2
    # Size of batches collected from each worker.
    custom_configuration['rollout_fragment_length'] = 200
    # Number of timesteps collected for each SGD round. This defines the size
    # of each SGD epoch.
    custom_configuration['train_batch_size'] = 4000
    # Total SGD batch size across all devices for SGD. This defines the
    # minibatch size within each epoch.
    custom_configuration['sgd_minibatch_size'] = 128
    # Whether to shuffle sequences in the batch when training (recommended).
    custom_configuration['shuffle_sequences'] = True
    # Number of SGD iterations in each outer loop (i.e., number of epochs to
    # execute per train batch).
    custom_configuration['num_sgd_iter'] = 30
    # Share layers for value function. If you set this to True, it's important
    # to tune vf_loss_coeff.
    custom_configuration['vf_share_layers'] = False
    # Coefficient of the value function loss. IMPORTANT: you must tune this if
    # you set vf_share_layers: True.
    custom_configuration['vf_loss_coeff'] = 1.0
    # Coefficient of the entropy regularizer.
    custom_configuration['entropy_coeff'] = 0.0
    # Decay schedule for the entropy regularizer.
    custom_configuration['entropy_coeff_schedule'] = None
    # PPO clip parameter.
    custom_configuration['clip_param'] = 0.3
    # Clip param for the value function. Note that this is sensitive to the
    # scale of the rewards. If your expected V is large, increase this.
    custom_configuration['vf_clip_param'] = 10.0
    # If specified, clip the global norm of gradients by this amount.
    custom_configuration['grad_clip'] = None
    # Target value for KL divergence.
    custom_configuration['kl_target'] = 0.01
    # Which observation filter to apply to the observation.
    custom_configuration['observation_filter'] = "NoFilter"
    # Uses the sync samples optimizer instead of the multi-gpu one. This is
    # usually slower, but you might want to try it if you run into issues with
    # the default optimizer.
    custom_configuration['simple_optimizer'] = False
    # Whether to fake GPUs (using CPUs).
    # Set this to True for debugging on non-GPU machines (set `num_gpus` > 0).
    custom_configuration['_fake_gpus'] = False

    # === MODEL ===
    custom_configuration['model']['use_lstm'] = False

    # === Evaluation Settings ===
    # Evaluate with every `evaluation_interval` training iterations.
    # The evaluation stats will be reported under the "evaluation" metric key.
    # Note that evaluation is currently not parallelized, and that for Ape-X
    # metrics are already only reported for the lowest epsilon workers.
    custom_configuration['evaluation_interval'] = 1
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
    custom_configuration['evaluation_num_workers'] = 2
    # Customize the evaluation method. This must be a function of signature
    # (trainer: Trainer, eval_workers: WorkerSet) -> metrics: dict. See the
    # Trainer._evaluate() method to see the default implementation. The
    # trainer guarantees all eval workers have the latest policy state before
    # this function is called.
    custom_configuration['custom_eval_function'] = None #custom_eval_function

    return custom_configuration
