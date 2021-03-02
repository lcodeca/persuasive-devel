#!/usr/bin/env python3

import sys

import matplotlib.pyplot as plt
import numpy as np

wanted_arrival = 32400

late_weight = 2.0
waiting_weight = 1.0

slotted_time = 15

max_penalty = 39600 / 60 # in minutes
max_penalty /= slotted_time # slotted time
max_penalty *= late_weight

arrival_buffer = wanted_arrival - (slotted_time * 60)

def get_reward(arrival):
    #### ERRORS
    if arrival is None:
        return 0 - max_penalty

    #### TOO LATE
    if wanted_arrival < arrival:
        penalty = arrival - wanted_arrival
        penalty /= 60 # in minutes
        penalty /= slotted_time # slotted time
        penalty += 1 # late is always bad
        return 0 - penalty * late_weight

    #### TOO EARLY
    if arrival_buffer > arrival:
        penalty = wanted_arrival - arrival
        penalty /= 60 # in minutes
        penalty /= slotted_time # slotted time
        return 0 - penalty * waiting_weight

    #### ON TIME
    return 1

agents_arrival = range(6*3600, 11*3600)

rewards = []
for _arrival in agents_arrival:
    rewards.append(get_reward(_arrival))

plt.hlines(0, 6*3600, 11*3600, linestyles='dotted', linewidth=2, color='orange')
plt.vlines(wanted_arrival, min(rewards)-1, max(rewards)+1, linestyles='dashed', linewidth=2, color='b', label='Wanted Arrival')
plt.vlines(arrival_buffer, min(rewards)-1, max(rewards)+1, linestyles='dashed', linewidth=2, color='g', label='Arrival Buffer')
plt.plot(agents_arrival, rewards, linewidth=2, color='r', label='Reward')
plt.title('Reward')
plt.xlabel('Time [s]')
plt.ylabel('Reward [#]')
plt.legend()
plt.savefig('reward.png')
plt.show()
plt.close()
# sys.exit(666)
