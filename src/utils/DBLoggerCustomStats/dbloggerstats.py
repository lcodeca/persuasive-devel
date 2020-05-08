#!/usr/bin/env python3

""" Process the DBLogger directory structure """

import json
import logging
import os
import re

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

class DBLoggerStats():
    """ Base class used to retrieve data from the DBLogger directory structure. """
    def __init__(self, directory):
        self.dir = directory

    @staticmethod
    def alphanumeric_sort(iterable):
        """
        Sorts the given iterable in the way that is expected.
        See: https://stackoverflow.com/questions/2669059/how-to-sort-alpha-numeric-set-in-python
        """
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(iterable, key=alphanum_key)

    def get_training_components(self, training):
        """ Retrieve the content from the training directory. """
        agents = list()
        episodes = list()
        files = list()
        for thing in os.listdir(os.path.join(self.dir, training)):
            if 'agent' in thing:
                agents.append(thing)
            elif 'episode' in thing:
                episodes.append(thing)
            else:
                files.append(thing)
        return self.alphanumeric_sort(agents), self.alphanumeric_sort(episodes), files

    def get_reward(self, training, agent):
        """ Retrieve 'agent_reward' from stats.json """
        fname = os.path.join(self.dir, training, agent, 'stats.json')
        vals = {}
        with open(fname, 'r') as jsonfile:
            vals = json.load(jsonfile)
            LOGGER.debug('%s ==> %s', fname, str(vals))
        return float(vals['agent_reward'])

    def get_timesteps_this_iter(self, training):
        """ Retrieve 'timesteps_this_iter' from aggregated-values.json """
        fname = os.path.join(self.dir, training, 'aggregated-values.json')
        vals = {}
        with open(fname, 'r') as jsonfile:
            vals = json.load(jsonfile)
            LOGGER.debug('%s ==> %s', fname, str(vals))
        return int(vals['timesteps_this_iter'])

    def get_last_action(self, training, episode, agent):
        """ Retrieve the last action from learning-sequence.json """
        fname = os.path.join(self.dir, training, episode, agent, 'learning-sequence.json')
        vals = {}
        with open(fname, 'r') as jsonfile:
            vals = json.load(jsonfile)
            LOGGER.debug('%s ==> %s', fname, str(vals))
        _, action, _, _ = vals[-1]
        return action

    def get_best_actions(self, training, agent):
        """ Load 'best-action.json' """
        fname = os.path.join(self.dir, training, agent, 'best-action.json')
        best_actions = {}
        with open(fname, 'r') as jsonfile:
            best_actions = json.load(jsonfile)
            LOGGER.debug('%s ==> %s', fname, str(best_actions))
        return best_actions

####################################################################################################
