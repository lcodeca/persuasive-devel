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

    def get_learning_sequence(self, training, episode, agent):
        """ Retrieve the learning-sequence.json """
        fname = os.path.join(self.dir, training, episode, agent, 'learning-sequence.json')
        sequence = {}
        with open(fname, 'r') as jsonfile:
            sequence = json.load(jsonfile)
            LOGGER.debug('%s ==> %s', fname, str(sequence))
        return sequence

    def get_last_action(self, training, episode, agent):
        """ Retrieve the last action from learning-sequence.json """
        fname = os.path.join(self.dir, training, episode, agent, 'learning-sequence.json')
        vals = {}
        with open(fname, 'r') as jsonfile:
            vals = json.load(jsonfile)
            LOGGER.debug('%s ==> %s', fname, str(vals))
        _, action, _, _ = vals[-1]
        return action

    def get_last_reward(self, training, episode, agent):
        """ Retrieve the last reward from learning-sequence.json """
        fname = os.path.join(self.dir, training, episode, agent, 'learning-sequence.json')
        vals = {}
        with open(fname, 'r') as jsonfile:
            vals = json.load(jsonfile)
            LOGGER.debug('%s ==> %s', fname, str(vals))
        _, _, _, reward = vals[-1]
        return reward

    def get_info(self, training, episode, agent):
        """ Retrieve the info.json """
        fname = os.path.join(self.dir, training, episode, agent, 'info.json')
        info = {}
        with open(fname, 'r') as jsonfile:
            info = json.load(jsonfile)
            LOGGER.debug('%s ==> %s', fname, str(info))
        return info

    def get_best_actions(self, training, agent):
        """ Load 'best-action.json' """
        fname = os.path.join(self.dir, training, agent, 'best-action.json')
        best_actions = {}
        with open(fname, 'r') as jsonfile:
            best_actions = json.load(jsonfile)
            LOGGER.debug('%s ==> %s', fname, str(best_actions))
        return best_actions

    def get_state_action_counter(self, training, agent):
        """ Load 'state-action-counter.json' """
        fname = os.path.join(self.dir, training, agent, 'state-action-counter.json')
        state_action_counter = {}
        with open(fname, 'r') as jsonfile:
            state_action_counter = json.load(jsonfile)
            LOGGER.debug('%s ==> %s', fname, str(state_action_counter))
        return state_action_counter

    def get_qtable(self, training, agent):
        """ Load 'qtable.json' """
        fname = os.path.join(self.dir, training, agent, 'qtable.json')
        qtable = {}
        with open(fname, 'r') as jsonfile:
            qtable = json.load(jsonfile)
            LOGGER.debug('%s ==> %s', fname, str(qtable))
        return qtable

####################################################################################################
