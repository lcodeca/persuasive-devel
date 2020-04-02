#!/usr/bin/env python3

""" PPO Configuration """

def ppo_conf(tr_steps=1, debug_folder=None):
    """
        https://github.com/ray-project/ray/blob/master/rllib/agents/trainer.py#L42
        https://github.com/ray-project/ray/blob/master/rllib/agents/ppo/ppo.py#L18
    """
    return {
        'batch_mode': 'complete_episodes',
        'ignore_worker_failures': True,
        'no_done_at_end': True,
        'simple_optimizer': False,

        'num_cpus_for_driver': 1,
        'num_cpus_per_worker': 1,
        'num_envs_per_worker': 1,
        'num_gpus': 1,
        'num_gpus_per_worker': 1,
        'num_workers': 1,

        'sample_batch_size': tr_steps,
        'rollout_fragment_length': tr_steps,
        'train_batch_size': tr_steps,
        'sgd_minibatch_size': round(tr_steps / 2),      # Minibatch size must be <= train batch size

        'log_level': 'INFO',
        # Whether to write episode stats and videos to the agent log dir
        'monitor': True,
        # Specify where experiences should be saved
        'output': debug_folder,

        # Enable TF eager execution (TF policies only).
        "eager": False,
        # Enable tracing in eager mode. This greatly improves performance, but
        # makes it slightly harder to debug since Python code won't be evaluated
        # after the initial eager pass.
        "eager_tracing": False,
    }
