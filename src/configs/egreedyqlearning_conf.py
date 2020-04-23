#!/usr/bin/env python3

""" Epsilon Greedy Q-Learning Configuration """

def egreedy_qlearning_conf(tr_steps=1, debug_folder=None):
    """
        https://github.com/ray-project/ray/blob/master/rllib/agents/trainer.py#L42
        https://github.com/ray-project/ray/blob/master/rllib/agents/ppo/ppo.py#L18
    """
    return {
        # ---- Debugging ----
        # Whether to write episode stats and videos to the agent log dir
        'monitor': True,
        'log_level': 'INFO',
        'ignore_worker_failures': True,

        # ---- Environment ----
        'no_done_at_end': True,

        # ---- Resources ----
        'num_workers': 1,
        'num_gpus': 1,
        'num_cpus_per_worker': 1,
        'num_gpus_per_worker': 1,

        # ---- Execution ----
        'num_envs_per_worker': 1,
        'rollout_fragment_length': tr_steps,
        'train_batch_size': tr_steps,
        'batch_mode': 'complete_episodes',
        'timesteps_per_iteration': 1,
        'seed': 42,

        # ---- Offline Datasets ----
        # Specify where experiences should be saved
        'output': debug_folder,
    }
