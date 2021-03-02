#!/usr/bin/env python3

import sys

import matplotlib.pyplot as plt
import numpy as np

for a_shape in range(1, 9):
    for mode_min in range(15, 60*3, 15):
        s = (np.random.pareto(a_shape, 1000) + 1) * mode_min
        count, bins, _ = plt.hist(s, 100, density=True)
        fit = a_shape * mode_min**a_shape / bins**(a_shape+1)
        plt.plot(bins, max(count)*fit/max(fit), linewidth=2, color='r')
        plt.vlines(np.mean(s), 0, max(count), linewidth=2, color='b', label='Mean')
        plt.vlines(np.median(s), 0, max(count), linewidth=2, color='g', label='Median')
        title = 'Shape {} - Mode (min) {} - Min: {} - Max: {}'.format(
            a_shape, mode_min, round(np.min(s), 2), round(np.max(s), 2))
        title += '\n Mean: {} - Std: {} - Median: {}'.format(
            round(np.mean(s), 2), round(np.std(s), 2), round(np.median(s), 2))
        print(title)
        plt.title(title)
        plt.legend()
        plt.savefig('pareto-png/{}_{}.png'.format(a_shape, mode_min))
        # plt.show()
        plt.close()
        # sys.exit(666)
