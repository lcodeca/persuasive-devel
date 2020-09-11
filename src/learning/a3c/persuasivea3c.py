#!/usr/bin/env python3

""" Persuasive implementation of A3C Tensorflow-based Policy & Trainer """

import collections
import logging
import numpy as np
from pprint import pprint, pformat

# Policy
import ray
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.policy.tf_policy import LearningRateSchedule
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.tf_ops import explained_variance, make_tf_callable

# Trainer
from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.execution.rollout_ops import AsyncGradients
from ray.rllib.execution.train_ops import ApplyGradients
from ray.rllib.execution.metric_ops import StandardMetricsReporting

# Callbacks
from typing import Dict
from ray import tune
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks

from utils.logger import set_logging

####################################################################################################

tf1, tf, tfv = try_import_tf()

logger = set_logging(__name__)

####################################################################################################

# Straignt from https://github.com/ray-project/ray/blob/releases/0.8.7/rllib/agents/a3c/a3c.py#L14
DEFAULT_CONFIG = with_common_config({
    # Should use a critic as a baseline (otherwise don't use value baseline;
    # required for using GAE).
    "use_critic": True,
    # If true, use the Generalized Advantage Estimator (GAE)
    # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
    "use_gae": True,
    # Size of rollout batch
    "rollout_fragment_length": 10,
    # GAE(gamma) parameter
    "lambda": 1.0,
    # Max global norm for each gradient calculated by worker
    "grad_clip": 40.0,
    # Learning rate
    "lr": 0.0001,
    # Learning rate schedule
    "lr_schedule": None,
    # Value Function Loss coefficient
    "vf_loss_coeff": 0.5,
    # Entropy coefficient
    "entropy_coeff": 0.01,
    # Min time per iteration
    "min_iter_time_s": 5,
    # Workers sample async. Note that this increases the effective
    # rollout_fragment_length by up to 5x due to async buffering of batches.
    "sample_async": True,
})

# ##################################################################################################

# Persuasive implementation of A3C Tensorflow-based Policy
# See:
#         https://ray.readthedocs.io/en/latest/rllib-concepts.html#policies
#         https://github.com/ray-project/ray/blob/releases/0.8.7/rllib/agents/a3c/a3c_tf_policy.py

# ##################################################################################################

class PersuasiveA3CLoss:
    def __init__(self,
                 action_dist,
                 actions,
                 advantages,
                 v_target,
                 vf,
                 vf_loss_coeff=0.5,
                 entropy_coeff=0.01):
        log_prob = action_dist.logp(actions)

        # The "policy gradients" loss
        self.pi_loss = -tf.reduce_sum(log_prob * advantages)

        delta = vf - v_target
        self.vf_loss = 0.5 * tf.reduce_sum(tf.math.square(delta))
        self.entropy = tf.reduce_sum(action_dist.entropy())
        self.total_loss = (self.pi_loss + self.vf_loss * vf_loss_coeff -
                           self.entropy * entropy_coeff)

def actor_critic_loss(policy, model, dist_class, train_batch):
    model_out, _ = model.from_batch(train_batch)
    action_dist = dist_class(model_out, model)
    policy.loss = PersuasiveA3CLoss(
        action_dist, train_batch[SampleBatch.ACTIONS],
        train_batch[Postprocessing.ADVANTAGES],
        train_batch[Postprocessing.VALUE_TARGETS],
        model.value_function(),
        policy.config["vf_loss_coeff"],
        policy.config["entropy_coeff"])
    return policy.loss.total_loss

def postprocess_advantages(policy,
                           sample_batch,
                           other_agent_batches=None,
                           episode=None):
    completed = sample_batch[SampleBatch.DONES][-1]
    if completed:
        last_r = 0.0
    else:
        next_state = []
        for i in range(policy.num_state_tensors()):
            next_state.append(sample_batch["state_out_{}".format(i)][-1])
        last_r = policy._value(sample_batch[SampleBatch.NEXT_OBS][-1],
                               sample_batch[SampleBatch.ACTIONS][-1],
                               sample_batch[SampleBatch.REWARDS][-1],
                               *next_state)
    return compute_advantages(
        sample_batch, last_r, policy.config["gamma"], policy.config["lambda"],
        policy.config["use_gae"], policy.config["use_critic"])


def add_value_function_fetch(policy):
    return {SampleBatch.VF_PREDS: policy.model.value_function()}

class ValueNetworkMixin:
    def __init__(self):
        @make_tf_callable(self.get_session())
        def value(ob, prev_action, prev_reward, *state):
            model_out, _ = self.model({
                SampleBatch.CUR_OBS: tf.convert_to_tensor([ob]),
                SampleBatch.PREV_ACTIONS: tf.convert_to_tensor([prev_action]),
                SampleBatch.PREV_REWARDS: tf.convert_to_tensor([prev_reward]),
                "is_training": tf.convert_to_tensor(False),
            }, [tf.convert_to_tensor([s]) for s in state],
                                      tf.convert_to_tensor([1]))
            return self.model.value_function()[0]

        self._value = value

def stats(policy, train_batch):
    return {
        "cur_lr": tf.cast(policy.cur_lr, tf.float64),
        "policy_loss": policy.loss.pi_loss,
        "policy_entropy": policy.loss.entropy,
        "var_gnorm": tf.linalg.global_norm(
            list(policy.model.trainable_variables())),
        "vf_loss": policy.loss.vf_loss,
    }

def grad_stats(policy, train_batch, grads):
    return {
        "grad_gnorm": tf.linalg.global_norm(grads),
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy.model.value_function()),
    }

def clip_gradients(policy, optimizer, loss):
    grads_and_vars = optimizer.compute_gradients(
        loss, policy.model.trainable_variables())
    grads = [g for (g, v) in grads_and_vars]
    grads, _ = tf.clip_by_global_norm(grads, policy.config["grad_clip"])
    clipped_grads = list(zip(grads, policy.model.trainable_variables()))
    return clipped_grads

def setup_mixins(policy, obs_space, action_space, config):
    ValueNetworkMixin.__init__(policy)
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])

####################################################################################################

PersuasiveA3CTFPolicy = build_tf_policy(
    name="PersuasiveA3CTFPolicy",
    get_default_config=lambda: DEFAULT_CONFIG,
    loss_fn=actor_critic_loss,
    stats_fn=stats,
    grad_stats_fn=grad_stats,
    gradients_fn=clip_gradients,
    postprocess_fn=postprocess_advantages,
    extra_action_fetches_fn=add_value_function_fetch,
    before_loss_init=setup_mixins,
    mixins=[ValueNetworkMixin, LearningRateSchedule])

####################################################################################################

# ##################################################################################################

#     Persuasive implementation of A3C Tensorflow-based Trainer
#     See:
#         https://ray.readthedocs.io/en/latest/rllib-concepts.html#trainers
#         https://github.com/ray-project/ray/blob/releases/0.8.7/rllib/agents/a3c/a3c.py

# ##################################################################################################

def get_policy_class(config):
    return PersuasiveA3CTFPolicy

def validate_config(config):
    if config["entropy_coeff"] < 0:
        raise DeprecationWarning("entropy_coeff must be >= 0")
    if config["sample_async"] and config["framework"] == "torch":
        config["sample_async"] = False
        logger.warning("`sample_async=True` is not supported for PyTorch! "
                       "Multithreading can lead to crashes.")

def execution_plan(workers, config):
    # For A3C, compute policy gradients remotely on the rollout workers.
    grads = AsyncGradients(workers)

    # Apply the gradients as they arrive. We set update_all to False so that
    # only the worker sending the gradient is updated with new weights.
    train_op = grads.for_each(ApplyGradients(workers, update_all=False))

    return StandardMetricsReporting(train_op, workers, config)

####################################################################################################

PersuasiveA3CTrainer = build_trainer(
    name="PersuasiveA3C",
    default_config=DEFAULT_CONFIG,
    default_policy=A3CTFPolicy,
    get_policy_class=get_policy_class,
    validate_config=validate_config,
    execution_plan=execution_plan)

####################################################################################################

# ##################################################################################################

#     Persuasive callbacks implementation
#     See:
#         https://docs.ray.io/en/latest/rllib-training.html#callbacks-and-custom-metrics
#         https://github.com/ray-project/ray/blob/releases/0.8.7/rllib/agents/callbacks.py
#         https://github.com/ray-project/ray/blob/releases/0.8.7/rllib/evaluation/episode.py
#         https://github.com/ray-project/ray/blob/releases/0.8.7/rllib/examples/custom_metrics_and_callbacks.py

# ##################################################################################################

class PersuasiveCallbacks(DefaultCallbacks):
    def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, **kwargs):
        episode.custom_metrics = {
            'episode_average_departure': [],
            'episode_average_arrival': [],
            'episode_average_wait': [],
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
        departure = []
        arrival = []
        wait = []
        for info in episode._agent_to_last_info.values():
            if np.isnan(info['departure']):
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

    # def on_train_result(self, trainer, result: dict, **kwargs):
    #     print("trainer.train() result: {} -> {} episodes".format(
    #         trainer, result["episodes_this_iter"]))
