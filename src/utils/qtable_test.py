#!/usr/bin/env python3

""" Q-Table implementation (w tests) """

import collections
import unittest

import dill

from qtable import QTable

class QTableTest(unittest.TestCase):
    """ Unit test for QTable """

    default = 0.0
    actions = [0, 1, 2]

    def test_state_to_key(self):
        """ Test _state_to_key(key) function. """
        qtable = QTable(self.actions)
        state1 = collections.OrderedDict()
        state1['from'] = 1
        state1['to'] = 2
        state1['rank'] = [0, 1, 2]

        state2 = collections.OrderedDict()
        state2['from'] = 1
        state2['to'] = 2
        state2['rank'] = [0, 2, 1]

        self.assertNotEqual(qtable._state_to_key(state1), qtable._state_to_key(state2))

    def test_get_without_init(self):
        """ Test qtable[state] if state not exists."""
        qtable = QTable(self.actions)
        state = collections.OrderedDict()
        state['from'] = 1
        state['to'] = 2
        state['rank'] = [0, 1, 2]

        wanted = {
            0: self.default,
            1: self.default,
            2: self.default,
        }
        self.assertEqual(wanted, qtable[state])

    def test_get_with_init(self):
        """ Test qtable[state] if state not exists."""
        qtable = QTable(self.actions)
        state = collections.OrderedDict()
        state['from'] = 1
        state['to'] = 2
        state['rank'] = [0, 1, 2]
        wanted = {
            0: self.default,
            1: self.default,
            2: self.default,
        }
        qtable[state] = wanted
        self.assertEqual(wanted, qtable[state])

    def test_set_without_init(self):
        """ Test qtable[state][action] = var if state not exists."""
        qtable = QTable(self.actions)
        state = collections.OrderedDict()
        state['from'] = 1
        state['to'] = 2
        state['rank'] = [0, 2, 1]
        qtable[state][0] = 1.0
        changed = {
            0: 1.0,
            1: self.default,
            2: self.default,
        }
        self.assertEqual(changed, qtable[state])

    def test_set_with_init(self):
        """ Test qtable[state][action] = var if state not exists."""
        qtable = QTable(self.actions)
        state = collections.OrderedDict()
        state['from'] = 1
        state['to'] = 2
        state['rank'] = [0, 1, 2]
        _ = qtable[state]

        changed = {
            0: 1.0,
            1: self.default,
            2: self.default,
        }
        qtable[state][0] = 1.0
        self.assertEqual(changed, qtable[state])

    def test_max_without_init(self):
        """ Test max(key) function. """
        qtable = QTable(self.actions)
        state = collections.OrderedDict()
        state['from'] = 1
        state['to'] = 2
        state['rank'] = [0, 1, 2]
        self.assertEqual(self.default, qtable.max(state))

    def test_max_with_init(self):
        """ Test max(key) function. """
        qtable = QTable(self.actions)
        state = collections.OrderedDict()
        state['from'] = 1
        state['to'] = 2
        state['rank'] = [0, 1, 2]
        qtable[state][0] = 1.0
        self.assertEqual(1.0, qtable.max(state))

    def test_argmax_without_init(self):
        """ Test max(key) function. """
        qtable = QTable(self.actions)
        state = collections.OrderedDict()
        state['from'] = 1
        state['to'] = 2
        state['rank'] = [0, 1, 2]
        self.assertTrue(qtable.argmax(state) in self.actions)

    def test_argmax_with_init(self):
        """ Test max(key) function. """
        qtable = QTable(self.actions)
        state = collections.OrderedDict()
        state['from'] = 1
        state['to'] = 2
        state['rank'] = [0, 1, 2]
        qtable[state][0] = 1.0
        self.assertEqual(0, qtable.argmax(state))

    def test_argmax_with_parity(self):
        """ Test max(key) function. """
        qtable = QTable(self.actions)
        state = collections.OrderedDict()
        state['from'] = 1
        state['to'] = 2
        state['rank'] = [0, 1, 2]
        qtable[state][0] = 1.0
        qtable[state][1] = 1.0
        self.assertTrue(qtable.argmax(state) in [0, 1])

    def test_dill(self):
        """ Test the dillability of the class. """
        qtable = QTable(self.actions)

        state1 = collections.OrderedDict()
        state1['from'] = 1
        state1['to'] = 2
        state1['rank'] = [0, 1, 2]

        state2 = collections.OrderedDict()
        state2['from'] = 1
        state2['to'] = 2
        state2['rank'] = [0, 2, 1]

        # create the states
        _ = qtable[state1]
        _ = qtable[state2]

        wanted = str(qtable)
        test = str(dill.loads(dill.dumps(qtable)))
        self.assertEqual(wanted, test)

if __name__ == '__main__':
    unittest.main()
