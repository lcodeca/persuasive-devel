#!/usr/bin/env python3

""" Persuasive DQN Configuration """

from pprint import pprint

import ray
from ray.rllib.evaluation.metrics import collect_episodes, collect_metrics, summarize_episodes
from ray.rllib.models import ModelCatalog

import ray.rllib.agents.dqn as dqn

from learning.a3c.persuasivea3c import PersuasiveCallbacks
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

def persuasive_dqn_conf(rollout_size=10,
                        agents=100,
                        debug_folder=None,
                        eval_folder=None,
                        alpha=5e-4,
                        gamma=0.99):
    """
        https://github.com/ray-project/ray/blob/releases/0.8.7/rllib/agents/trainer.py#L44
        https://github.com/ray-project/ray/blob/releases/0.8.7/rllib/agents/dqn/dqn.py#L21
        https://github.com/ray-project/ray/blob/releases/0.8.7/rllib/models/catalog.py#L37
    """

    # ModelCatalog.register_custom_model('custom_rrn', RNNModel)
    ModelCatalog.register_custom_action_dist(
        'custom_action_distribution', PersuasiveActionDistribution)

    custom_configuration = dqn.DEFAULT_CONFIG.copy()

    custom_configuration['collect_metrics_timeout'] = 86400 # a day
    custom_configuration['framework'] = 'tf'
    custom_configuration['ignore_worker_failures'] = True
    custom_configuration['log_level'] = 'WARN'
    custom_configuration['monitor'] = True
    custom_configuration['num_cpus_for_driver'] = 1
    custom_configuration['num_cpus_per_worker'] = 1
    custom_configuration['num_envs_per_worker'] = 1
    custom_configuration['output'] = debug_folder
    custom_configuration['remote_env_batch_wait_ms'] = 1000
    custom_configuration['remote_worker_envs'] = False
    custom_configuration['seed'] = 42

    # === Parallelism ===
    # Number of workers for collecting samples with. This only makes sense
    # to increase if your environment is particularly slow to sample, or if
    # you"re using the Async or Ape-X optimizers.
    custom_configuration['num_workers'] = 0
    custom_configuration['num_gpus_per_worker'] = 1
    # Whether to compute priorities on workers.
    custom_configuration['worker_side_prioritization'] = False
    # Prevent iterations from going lower than this time span
    custom_configuration['min_iter_time_s'] = 1

    # === Environment Settings ===
    custom_configuration['batch_mode'] = 'complete_episodes'
    custom_configuration['callbacks'] = PersuasiveCallbacks
    custom_configuration['gamma'] = gamma
    custom_configuration['lr'] = alpha
    custom_configuration['no_done_at_end'] = False
    # Update the replay buffer with this many samples at once. Note that
    # this setting applies per-worker if num_workers > 1.
    custom_configuration['rollout_fragment_length'] = rollout_size
    # Size of a batch sampled from replay buffer for training. Note that
    # if async_updates is set, then each worker returns gradients for a
    # batch of this size.
    custom_configuration['train_batch_size'] = rollout_size * agents
    # If positive, input batches will be shuffled via a sliding window buffer
    # of this number of batches. Use this if the input data is not in random
    # enough order. Input is delayed until the shuffle buffer is filled.
    custom_configuration['shuffle_buffer_size'] = rollout_size * agents
    # Minimum env steps to optimize for per train call. This value does
    # not affect learning, only the length of train iterations.
    custom_configuration['timesteps_per_iteration'] = agents
    # How many steps of the model to sample before learning starts.
    custom_configuration['learning_starts'] = rollout_size * agents


    # === Exploration Settings ===
    # https://github.com/ray-project/ray/blob/releases/0.8.7/rllib/utils/exploration/epsilon_greedy.py
    custom_configuration['exploration_config']['type'] = 'EpsilonGreedy'
    custom_configuration['exploration_config']['initial_epsilon'] = 1.0
    custom_configuration['exploration_config']['final_epsilon'] = 0.02
    custom_configuration['exploration_config']['epsilon_timesteps'] = 10000
    # Name of a custom action distribution to use.
    # See: https://docs.ray.io/en/releases-0.8.7/rllib-models.html#custom-action-distributions
    custom_configuration['model']['custom_action_dist'] = 'custom_action_distribution'

    # === DQN Model Settings ===
    # Update the target network every `target_network_update_freq` steps.
    custom_configuration['target_network_update_freq'] = agents # every agent should have done at least 1 action

    # === Replay buffer ===
    # Size of the replay buffer. Note that if async_updates is set, then
    # each worker will have a replay buffer of this size.
    custom_configuration['buffer_size'] = 10000 # 50000
    # If True prioritized replay buffer will be used.
    custom_configuration['prioritized_replay'] = False
    # Alpha parameter for prioritized replay buffer.
    custom_configuration['prioritized_replay_alpha'] = 0.6
    # Beta parameter for sampling from prioritized replay buffer.
    custom_configuration['prioritized_replay_beta'] = 0.4
    # Final value of beta (by default, we use constant beta=0.4).
    custom_configuration['final_prioritized_replay_beta'] = 0.4
    # Time steps over which the beta parameter is annealed.
    custom_configuration['prioritized_replay_beta_annealing_timesteps'] = 20000
    # Epsilon to add to the TD errors when updating priorities.
    custom_configuration['prioritized_replay_eps'] = 1e-6
    # Whether to LZ4 compress observations
    custom_configuration['compress_observations'] = False
    # Callback to run before learning on a multi-agent batch of experiences.
    custom_configuration['before_learn_on_batch'] = None
    # If set, this will fix the ratio of replayed from a buffer and learned on
    # timesteps to sampled from an environment and stored in the replay buffer
    # timesteps. Otherwise, the replay will proceed at the native ratio
    # determined by (train_batch_size / rollout_fragment_length).
    custom_configuration['training_intensity'] = None

    # === Optimization ===
    # Adam epsilon hyper parameter
    custom_configuration['adam_epsilon'] = 1e-8
    # If not None, clip gradients during optimization at this value
    custom_configuration['grad_clip'] = 40

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
    custom_configuration['evaluation_num_workers'] = 1
    # Customize the evaluation method. This must be a function of signature
    # (trainer: Trainer, eval_workers: WorkerSet) -> metrics: dict. See the
    # Trainer._evaluate() method to see the default implementation. The
    # trainer guarantees all eval workers have the latest policy state before
    # this function is called.
    custom_configuration['custom_eval_function'] = None #custom_eval_function

    return custom_configuration
