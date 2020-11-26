
#!/usr/bin/env python3

"""
Persuasive callbacks implementation
See:
    https://docs.ray.io/en/latest/rllib-training.html#callbacks-and-custom-metrics
    https://github.com/ray-project/ray/blob/releases/1.0.0/rllib/agents/callbacks.py
    https://github.com/ray-project/ray/blob/releases/1.0.0/rllib/evaluation/episode.py
    https://github.com/ray-project/ray/blob/releases/1.0.0/rllib/examples/custom_metrics_and_callbacks.py
"""

import sys

from typing import Dict

import numpy as np

from ray import tune
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks

from utils.logger import set_logging

####################################################################################################

logger = set_logging(__name__)

####################################################################################################

class PersuasiveCallbacks(DefaultCallbacks):
    def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, **kwargs):
        episode.custom_metrics = {
            'episode_average_departure': [],
            'episode_average_arrival': [],
            'episode_average_wait': [],
            'episode_missing_agents': [],
            'episode_on_time_agents': [],
            'episode_total_reward': [],
        }
        episode.hist_data = {
            'info_by_agent': [],
            'rewards_by_agent': [],
            'last_action_by_agent': [],
        }

    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       **kwargs):
        episode.hist_data['info_by_agent'].append(episode._agent_to_last_info)
        episode.hist_data['rewards_by_agent'].append(episode._agent_reward_history)
        episode.hist_data['last_action_by_agent'].append(episode._agent_to_last_action)

        missing = 0
        departure = []
        arrival = []
        wait = []
        for info in episode._agent_to_last_info.values():
            if np.isnan(info['departure']):
                missing += 1
                continue
            departure.append(info['departure'])
            if np.isnan(info['arrival']):
                continue
            arrival.append(info['arrival'])
            if np.isnan(info['wait']):
                continue
            wait.append(info['wait'])
        episode.custom_metrics['episode_average_departure'].append(np.mean(departure))
        episode.custom_metrics['episode_average_arrival'].append(np.mean(arrival))
        episode.custom_metrics['episode_average_wait'].append(np.mean(wait))
        episode.custom_metrics['episode_missing_agents'].append(missing)

        on_time = 0
        total_reward = 0
        for reward in episode._agent_reward_history.values():
            _sum = np.sum(reward)
            total_reward += _sum
            if _sum >= 1:
                on_time += 1
        episode.custom_metrics['episode_on_time_agents'].append(on_time)
        episode.custom_metrics['episode_total_reward'].append(total_reward)

    # def on_train_result(self, trainer, result: dict, **kwargs):
    #     print("trainer.train() result: {} -> {} episodes".format(
    #         trainer, result["episodes_this_iter"]))
