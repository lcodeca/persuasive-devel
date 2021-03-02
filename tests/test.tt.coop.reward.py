#!/usr/bin/env python3

import sys

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

wanted_arrival = 32400

late_weight = 2.0
waiting_weight = 1.0

slotted_arrival_time = 15
slotted_travel_time = 5

max_travel_time = 4 * 3600 / 60 # in minutes
max_travel_time /= slotted_arrival_time # slotted time
max_travel_time *= late_weight

arrival_buffer = wanted_arrival - (slotted_arrival_time * 60)

def get_reward(arrival, travel_time, on_time):

    on_time_penalty = 1 / on_time

    #### ERRORS
    if arrival is None:
        return 0 - (max_travel_time * 2) * late_weight * on_time_penalty

    travel_time_penalty = travel_time / 60 # in minutes
    travel_time_penalty /= slotted_travel_time

    #### TOO LATE
    if wanted_arrival < arrival:
        penalty = arrival - wanted_arrival
        penalty /= 60 # in minutes
        penalty /= slotted_arrival_time # slotted time
        penalty += 1 # late is always bad
        return 0 - (travel_time_penalty + penalty) * late_weight * on_time_penalty

    #### TOO EARLY
    if arrival_buffer > arrival:
        penalty = wanted_arrival - arrival
        penalty /= 60 # in minutes
        penalty /= slotted_arrival_time # slotted time
        return 0 - (travel_time_penalty + penalty) * waiting_weight * on_time_penalty

    #### ON TIME
    return 1 - travel_time_penalty * on_time_penalty

agents_arrival = range(6 * 3600, 11 * 3600)
agents_travel = range(1 * 60, 3 * 3600, 3 * 60)
agents_on_time = range(0, 1001, 100)

for pos, _time in tqdm(enumerate(agents_travel)):
    min_reward = get_reward(None, _time, 1/1000.0)
    plt.hlines(0, 6*3600, 11*3600, linestyles='solid', color='black')
    plt.hlines(min_reward, 6*3600, 11*3600, linestyles='dotted', linewidth=2, color='red', label='Max Penalty')
    plt.vlines(wanted_arrival, min_reward, 1, linestyles='dashed', linewidth=2, color='blue', label='Wanted Arrival')
    plt.vlines(arrival_buffer, min_reward, 1, linestyles='dashed', linewidth=2, color='green', label='Arrival Buffer')

    for _on_time in agents_on_time:
        if _on_time <= 0:
            _on_time = 1
        _on_time = _on_time / 1000.0
        rewards = []
        for _arrival in agents_arrival:
            rewards.append(get_reward(_arrival, _time, _on_time))
        plt.plot(agents_arrival, rewards, label='On-Time Penalty {}'.format(_on_time))

    plt.title('Reward associated with {} min travel time.'.format(_time/60))
    plt.yscale('symlog')
    plt.xlabel('Time [s]')
    plt.ylabel('Reward [log]')
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig('reward_tt_coop/reward_{}.png'.format(pos), bbox_extra_artists=(lgd,), bbox_inches='tight')
    # plt.show()
    plt.close()
    # sys.exit(666)
