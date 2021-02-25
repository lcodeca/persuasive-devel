#!/usr/bin/env python3

"""
    Persuasive implementation of a
    custom model with action masking tied to the ownership of a vehicle.
    See:
    - https://docs.ray.io/en/master/rllib-models.html#variable-length-parametric-action-spaces
    - https://github.com/ray-project/ray/blob/releases/1.0.0/rllib/examples/models/parametric_actions_model.py

"""

import logging
import numpy as np

from gym.spaces import Box
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf

from utils.logger import set_logging

####################################################################################################

tf1, tf, tfv = try_import_tf()

logger = set_logging(__name__)

####################################################################################################

class OwnershipActionMaskingModel(FullyConnectedNetwork):
    """
    Parametric action model that handles the dot product and masking.
    This assumes the outputs are logits for a single Categorical action dist.
    """

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 true_obs_shape=(19, ),
                 action_embed_size=6,
                 **kw):
        # super(OwnershipActionMaskingModel, self).__init__(
        #     Box(-1, 1, shape=true_obs_shape),
        #     action_space, num_outputs, model_config, name, **kw)
        super(OwnershipActionMaskingModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, **kw)
        self.action_embed_model = FullyConnectedNetwork(
            Box(-1, 1, shape=true_obs_shape), action_space, action_embed_size,
            model_config, name + "_action_embed")
        self.register_variables(self.action_embed_model.variables())

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        avail_actions = input_dict["obs"]["avail_actions"]
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the predicted action embedding
        action_embed, _ = self.action_embed_model({
            "obs": input_dict["obs"]["obs"]
        })

        # model_out, self._value_out = self.base_model(input_dict["obs"]["obs"])

        # Expand the model output to [BATCH, 1, EMBED_SIZE]. Note that the
        # avail actions tensor is of shape [BATCH, MAX_ACTIONS, EMBED_SIZE].
        intent_vector = tf.expand_dims(action_embed, 1)

        # Batch dot product => shape of logits is [BATCH, MAX_ACTIONS].
        action_logits = tf.reduce_sum(avail_actions * intent_vector, axis=1)

        # Mask out invalid actions (use tf.float32.min for stability)
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()
