#!/usr/bin/env python3

"""
    Persuasive implementation of a custom action distribution.
    See:
    - https://docs.ray.io/en/releases-0.8.7/rllib-models.html#custom-action-distributions
    - https://github.com/ray-project/ray/blob/releases/0.8.6/rllib/models/action_dist.py
    - https://github.com/ray-project/ray/blob/releases/0.8.7/rllib/models/tf/tf_action_dist.py
"""

import logging
import numpy as np

from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.framework import try_import_tf

####################################################################################################

tf1, tf, tfv = try_import_tf()

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

####################################################################################################

class PersuasiveActionDistribution(TFActionDistribution):
    """ Persuasive custom distribution for discrete action spaces."""

    @DeveloperAPI
    def __init__(self, inputs, probabilities, model=None):
        super().__init__(inputs, model)
        self.probabilities = probabilities

    @override(ActionDistribution)
    def deterministic_sample(self):
        return tf.math.argmax(self.inputs, axis=1)

    @override(ActionDistribution)
    def logp(self, x):
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.inputs, labels=tf.cast(x, tf.int32))

    @override(ActionDistribution)
    def entropy(self):
        a0 = self.inputs - tf.reduce_max(self.inputs, axis=1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis=1)

    @override(ActionDistribution)
    def kl(self, other):
        a0 = self.inputs - tf.reduce_max(self.inputs, axis=1, keepdims=True)
        a1 = other.inputs - tf.reduce_max(other.inputs, axis=1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=1, keepdims=True)
        z1 = tf.reduce_sum(ea1, axis=1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(
            p0 * (a0 - tf.math.log(z0) - a1 + tf.math.log(z1)), axis=1)

    @override(TFActionDistribution)
    def _build_sample_op(self):
        print('\n\n\n\n')
        print(self.inputs)
        # probabilities = tf.Tensor([[0.9, 0.1]])
        # probabilities = tf.compat.v1.placeholder(tf.float32, shape=(None, 2))
        # logits = tf.math.log(probabilities)
        # print(probabilities)
        # print(logits)
        print('\n\n\n\n')
        categorical = tf.random.categorical(self.inputs, 1)
        # categorical2 = tf.random.categorical(logits, 1)
        print(categorical)
        # print(categorical2)
        squeeze = tf.squeeze(categorical, axis=1)
        # squeeze2 = tf.squeeze(categorical2, axis=1)
        print(squeeze)
        # print(squeeze2)
        print('\n\n\n\n')
        # print(squeeze2.numpy())
        return squeeze

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space, model_config):
        return action_space.n
