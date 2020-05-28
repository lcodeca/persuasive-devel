#!/usr/bin/env python3

""" Q-Table implementation. """

import collections
from copy import deepcopy
import logging
from pprint import pformat

from numpy.random import RandomState

####################################################################################################

LOGGER = logging.getLogger(__name__)

####################################################################################################

class QTable(dict): # collections.OrderedDict # collections.defaultdict
    """
    Implements the functionality required by Q-Learning associated with access and generation
    of Q-Values.

    Initialization:
        table = QTable(actions, default_q-value)

    Access:
        - table[state]
            the state must be a collections.OrderedDict
            returns the dict { action: value, ... }
        - table[state] = value
            the state must be a collections.OrderedDict
            saves value in table[state]
        - table.max(state)
            the state must be a collections.OrderedDict
            returns the max among all the q-values associated with the actions
        - table.argmax(state)
            the state must be a collections.OrderedDict
            returns the action with the max q-value, or a random choice in case of parity.
    """

    def __init__(self, actions, default=0.0, seed=None):
        dict.__init__(self)
        LOGGER.debug('actions %s', pformat(actions))
        LOGGER.debug('default %f', default)
        LOGGER.debug('seed %s', pformat(seed))
        self._data = dict()
        self._default_value = default
        self._default_actions = set(actions)
        self._seed = seed
        self._key_to_state = dict()
        if seed:
            self._rndgen = RandomState(seed)
        else:
            self._rndgen = RandomState()

    def _generate_default_value(self):
        """ Return a default value. Handles the difference between the possible default objects """
        if callable(self._default_value):
            return self._default_value()
        return deepcopy(self._default_value)

    def _generate_default_state(self):
        """ Return the default state. """
        state = {action: self._generate_default_value() for action in self._default_actions}
        return state

    def _state_to_key(self, state):
        """
        Makes a consistent string from a dictionary, list, tuple or set to any level, that contains
        only other hashable types (including any lists, tuples, sets, and dictionaries).
        See: https://stackoverflow.com/a/8714242
        """
        key = pformat(state.items())
        self._key_to_state[key] = state
        return key

    def get_state_from_serialized_key(self, key, default=None):
        """
        Return the original state (collections.OrderedDict) associated with a serialized key 4
        or the default value (None) otherwise.
        """
        if key in self._key_to_state:
            return self._key_to_state[key]
        return default

    def _get_actual_key(self, key):
        """
        The key should be a collections.OrderedDict or it should be already stored in the dataset.
        """
        if not isinstance(key, collections.OrderedDict):
            if key in self._key_to_state:
                return key # it has been already serialized
            raise KeyError(
                'The state/key must be collections.OrderedDict or it must already be stored.',
                key)
        return self._state_to_key(key) # serialize key

    def __getitem__(self, key):
        """
        Given that the table must be difined for each state, when a state is requested for the first
        time, it is generated, inserted, and returned on-the-fly.
        """
        LOGGER.debug('Key %s', pformat(key))
        actual_key = self._get_actual_key(key)
        if actual_key not in self._data: # generate if necessary
            self._data[actual_key] = self._generate_default_state()
        return self._data[actual_key]

    def __setitem__(self, key, value):
        LOGGER.debug('Key %s', pformat(key))
        LOGGER.debug('Value %s', pformat(value))
        actual_key = self._get_actual_key(key)
        if self._default_actions != set(value.keys()):
            raise KeyError('The only actions allowed in the "value" are:', self._default_actions)
        self._data[actual_key] = value

    def max(self, key):
        """ [always defined] Returns the max among all the q-values associated with the actions. """
        LOGGER.debug('Key %s', pformat(key))
        actual_key = self._get_actual_key(key)
        if actual_key not in self._data: # generate if necessary
            LOGGER.debug('State created on the fly.')
            self._data[actual_key] = self._generate_default_state()
        max_val = None
        LOGGER.debug('-----------------------------------------')
        for _, value in self._data[actual_key].items():
            LOGGER.debug('%s', str(value))
            if max_val is None:
                max_val = value
            else:
                max_val = max(value, max_val)
        LOGGER.debug('============ %s ============', str(max_val))
        return max_val

    def maxactions(self, key):
        """ [always defined] Returns the set of action associated with the maximum value """
        LOGGER.debug('Key %s', pformat(key))
        actual_key = self._get_actual_key(key)
        if actual_key not in self._data: # generate if necessary
            self._data[actual_key] = self._generate_default_state()
        max_val = None
        max_keys = []
        for key, value in self._data[actual_key].items():
            LOGGER.debug(' ITEM: %s --> %s', str(key), pformat(value))
            LOGGER.debug(' Before --> max_val: %s - max_keys: %s', str(max_val), pformat(max_keys))
            if max_val is None:
                max_val = value
                max_keys = [key]
            elif max_val == value:
                max_keys.append(key)
            elif max_val < value:
                max_val = value
                max_keys = [key]
            LOGGER.debug(' After --> max_val: %s - max_keys: %s', str(max_val), pformat(max_keys))
        LOGGER.debug('ARGMAX: max_val: %s - max_keys: %s', str(max_val), pformat(max_keys))
        return set(max_keys)

    def argmax(self, key):
        """ [always defined] Returns the max among all the q-values associated with the actions. """
        max_keys = self.maxactions(key)
        return self._rndgen.choice(list(max_keys))

    def __str__(self):
        return pformat(self._data)

    def __repr__(self):
        return pformat(self._data)

    def get_flattened_dict(self):
        flattened_data = list()
        for state, item in self._data.items():
            for action, value in item.items():
                flattened_data.append((state, action, value))
        return flattened_data

    def get_items(self):
        return self._data.items()
