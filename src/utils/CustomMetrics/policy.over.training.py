#!/usr/bin/env python3

""" Process the RLLIB metrics_XYZ.json """

import argparse
import collections
import cProfile
import io
import json
import logging
import os
import pstats
import random
import sys
from pprint import pformat, pprint

import matplotlib
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as geometry
from tqdm import tqdm

from genericgraphmaker import GenericGraphMaker

# """ Import SUMO library """
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
    import sumolib
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

####################################################################################################

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

random.seed()

####################################################################################################

SMALL_SIZE = 20
MEDIUM_SIZE = SMALL_SIZE + 4
BIGGER_SIZE = MEDIUM_SIZE + 4

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

####################################################################################################

def _argument_parser():
    """ Argument parser for the stats parser. """
    parser = argparse.ArgumentParser(description='RLLIB & SUMO Statistics parser.')
    parser.add_argument(
        '--input-dir', required=True, type=str, help='Input JSONs directory.')
    parser.add_argument(
        '--output-dir', default='stats', help='Output aggregation & graphs directory.')
    parser.add_argument(
        '--profiler', dest='profiler', action='store_true', help='Enable cProfile.')
    parser.set_defaults(profiler=False)
    return parser.parse_args()

def _main():
    """ Process the RLLIB logs/result.json """

    config = _argument_parser()

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler = cProfile.Profile()
        profiler.enable()
    ## ========================              PROFILER              ======================== ##

    Policy(config.input_dir, config.output_dir).generate()
    logging.info('Done')

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler.disable()
        results = io.StringIO()
        pstats.Stats(profiler, stream=results).sort_stats('cumulative').print_stats(50)
        logging.info('Profiler: \n%s', pformat(results.getvalue()))
        print('Profiler: \n%s', results.getvalue())
    ## ========================              PROFILER              ======================== ##

####################################################################################################

class Policy(GenericGraphMaker):
    """ Process a SINGLE RLLIB logs/result.json file as a time series. """

    def __init__(self, input_dir, output_dir):
        _default = collections.defaultdict(dict)
        super().__init__(
            input_dir, output_dir,
            filename='policies.json',
            default=_default)
        self._action_to_mode = {
            0: 'wait',
            1: 'passenger',
            2: 'public',
            3: 'walk',
            4: 'bicycle',
            5: 'ptw',
            6: 'on-demand',
        }
        self._mode_to_color = {
            'wait': 'black',
            'passenger': 'red',
            'public': 'green',
            'walk': 'orange',
            'bicycle': 'blue',
            'ptw': 'purple',
            'on-demand': 'grey',
        }

    def _find_last_metric(self):
        return len(self._aggregated_dataset)

    def _aggregate_metrics(self, files):
        for filename in tqdm(files):
            with open(os.path.join(self._input_dir, filename), 'r') as jsonfile:
                complete = json.load(jsonfile)

                if 'action-to-mode' in complete['config']['env_config']['agent_init']:
                    action_to_mode = \
                        complete['config']['env_config']['agent_init']['action-to-mode']
                else:
                    action_to_mode = self._action_to_mode.copy()

                training_iteration = int(complete['training_iteration'])
                expected_arrival_s = complete['config']['env_config']['agent_init'][
                    'expected-arrival-time'][0]
                slots_m = complete['config']['env_config']['agent_init']['arrival-slots-min']

                ## LEARNING
                info_by_episode = complete['hist_stats']['info_by_agent']
                last_action_by_agent = complete['hist_stats']['last_action_by_agent']
                # rewards_by_agent = complete['hist_stats']['rewards_by_agent']
                pos = random.randrange(len(info_by_episode))
                episode = info_by_episode[pos]

                learning = {
                    'action-to-mode': action_to_mode,
                    'expected_arrival_s': expected_arrival_s,
                    'slots_m': slots_m,
                    'agents_num': len(episode),
                    'agents': list(),
                }

                mode_usage = collections.defaultdict(int)
                on_time = 0
                too_late = 0
                too_early = 0
                missing = 0
                for agent, info in episode.items():
                    mode_usage[last_action_by_agent[pos][agent]] += 1
                    if np.isnan(info['arrival']):
                        missing += 1
                    else:
                        tmp = {
                            'id': int(agent.split('_')[1]),
                            'start': info['init']['start']/3600,
                            'mode': last_action_by_agent[pos][agent],
                            'arrival': info['arrival']/3600.0,
                            'departure': info['departure']/3600.0,
                        }
                        learning['agents'].append(tmp)
                        if info['arrival'] > expected_arrival_s:
                            too_late += 1
                        elif info['arrival'] > (expected_arrival_s - slots_m * 60):
                            on_time += 1
                        else:
                            too_early += 1

                learning['mode_usage'] = mode_usage
                learning['on_time'] = on_time
                learning['too_late'] = too_late
                learning['too_early'] = too_early
                learning['missing'] = missing
                self._aggregated_dataset[training_iteration] = {
                    'learning': learning,
                }

                # EVALUATION
                if 'evaluation' in complete:
                    complete = complete['evaluation']
                    info_by_episode = complete['hist_stats']['info_by_agent']
                    last_action_by_agent = complete['hist_stats']['last_action_by_agent']
                    # rewards_by_agent = complete['hist_stats']['rewards_by_agent']
                    pos = random.randrange(len(info_by_episode))
                    episode = info_by_episode[pos]

                    evaluation = {
                        'action-to-mode': action_to_mode,
                        'expected_arrival_s': expected_arrival_s,
                        'slots_m': slots_m,
                        'agents_num': len(episode),
                        'agents': list(),
                    }

                    mode_usage = collections.defaultdict(int)
                    on_time = 0
                    too_late = 0
                    too_early = 0
                    missing = 0
                    for agent, info in episode.items():
                        mode_usage[last_action_by_agent[pos][agent]] += 1
                        if np.isnan(info['arrival']):
                            missing += 1
                        else:
                            tmp = {
                                'id': int(agent.split('_')[1]),
                                'start': info['init']['start']/3600,
                                'mode': last_action_by_agent[pos][agent],
                                'arrival': info['arrival']/3600.0,
                                'departure': info['departure']/3600.0,
                            }
                            evaluation['agents'].append(tmp)
                            if info['arrival'] > expected_arrival_s:
                                too_late += 1
                            elif info['arrival'] > (expected_arrival_s - slots_m * 60):
                                on_time += 1
                            else:
                                too_early += 1

                    evaluation['mode_usage'] = mode_usage
                    evaluation['on_time'] = on_time
                    evaluation['too_late'] = too_late
                    evaluation['too_early'] = too_early
                    evaluation['missing'] = missing
                    self._aggregated_dataset[training_iteration]['evaluation'] = evaluation

    def _perc(self, num, agents):
        return round(num / agents * 100.0, 1)

    def _generate_graphs(self):
        already_plotted = []
        for filename in os.listdir(self._output_dir):
            if 'svg' in filename:
                already_plotted.append(filename)

        for missing_plot in tqdm(range(len(already_plotted)+1, len(self._aggregated_dataset)+1)):
            ################################################################
            ##                     Setup the images
            ################################################################

            fig, axs = plt.subplots(1, 2, figsize=(25, 15), constrained_layout=True)
            fig.suptitle('Policy for training run {}'.format(missing_plot))

            # ################################################################

            current = self._aggregated_dataset[missing_plot]['learning']
            self._action_to_mode = {}
            for _action, _mode in current['action-to-mode'].items():
                self._action_to_mode[int(_action)] = _mode
            self._action_to_mode[0] = 'wait'

            # ############ LEARNIN
            axs[0].bar(current['expected_arrival_s']/3600, current['agents_num'],
                       width=0.05, color='r', align='center')
            axs[0].bar((current['expected_arrival_s'] - current['slots_m'] * 60) / 3600,
                       current['agents_num'], width=0.02, color='g', align='center')
            axs[0].set_xlim(6.0, 10.0)
            axs[0].set_ylim(-1, current['agents_num']+1)
            axs[0].set_xlabel('Time [h]')
            axs[0].set_ylabel('Agents [#]')

            for agent in current['agents']:
                y = [agent['id'], agent['id']]
                axs[0].plot(
                    [agent['start'], agent['departure']], y, color='black', alpha=0.5)
                axs[0].plot(
                    [agent['departure'], agent['arrival']], y,
                    color=self._mode_to_color[self._action_to_mode[agent['mode']]], alpha=0.9)

            title = 'Learning \n'
            title += 'early {} - on time {} - late {} - missing {} \n'.format(
                current['too_early'], current['on_time'], current['too_late'], current['missing'])
            title += 'early {}% - on time {}% - late {}% - missing {}% \n'.format(
                self._perc(current['too_early'], current['agents_num']),
                self._perc(current['on_time'], current['agents_num']),
                self._perc(current['too_late'], current['agents_num']),
                self._perc(current['missing'], current['agents_num']))
            axs[0].set_title(title)
            labels = []
            for action in sorted(self._action_to_mode):
                color = self._mode_to_color[self._action_to_mode[action]]
                label = '{} \n({}%)'.format(
                    self._action_to_mode[action],
                    self._perc(current['mode_usage'][action], current['agents_num']))
                labels.append(mpatches.Patch(color=color, label=label))
            axs[0].legend(handles=labels, loc='upper left')

            ############ EVALUATION
            if 'evaluation' in self._aggregated_dataset[missing_plot]:
                current = self._aggregated_dataset[missing_plot]['evaluation']

                axs[1].bar(current['expected_arrival_s']/3600, current['agents_num'],
                           width=0.05, color='r', align='center')
                axs[1].bar((current['expected_arrival_s'] - current['slots_m'] * 60) / 3600,
                           current['agents_num'], width=0.02, color='g', align='center')
                axs[1].set_xlim(6.0, 10.0)
                axs[1].set_ylim(-1, current['agents_num'])
                axs[1].set_xlabel('Time [h]')
                axs[1].set_ylabel('Agents [#]')

                for agent in current['agents']:
                    y = [agent['id'], agent['id']]
                    axs[1].plot(
                        [agent['start'], agent['departure']], y, color='black', alpha=0.5)
                    axs[1].plot(
                        [agent['departure'], agent['arrival']], y,
                        color=self._mode_to_color[self._action_to_mode[agent['mode']]], alpha=0.9)

                title = 'Evaluation \n'
                title += 'early {} - on time {} - late {} - missing {} \n'.format(
                    current['too_early'], current['on_time'], current['too_late'],
                    current['missing'])
                title += 'early {}% - on time {}% - late {}% - missing {}% \n'.format(
                    self._perc(current['too_early'], current['agents_num']),
                    self._perc(current['on_time'], current['agents_num']),
                    self._perc(current['too_late'], current['agents_num']),
                    self._perc(current['missing'], current['agents_num']))
                axs[1].set_title(title)
                labels = []
                for action in sorted(self._action_to_mode):
                    color = self._mode_to_color[self._action_to_mode[action]]
                    label = '{} \n({}%)'.format(
                        self._action_to_mode[action],
                        self._perc(current['mode_usage'][action], current['agents_num']))
                    labels.append(mpatches.Patch(color=color, label=label))
                axs[1].legend(handles=labels, loc='upper left')

            ################################################################
            fig.savefig('{}/policy.{}.svg'.format(self._output_dir, missing_plot),
                        dpi=300, transparent=False, bbox_inches='tight')
            # fig.savefig('{}.{}.png'.format(self.prefix, self.agent_name),
            #             dpi=300, transparent=False, bbox_inches='tight')
            # plt.show()
            matplotlib.pyplot.close('all')
            # sys.exit()
            ###############################################################
        #################################

####################################################################################################

if __name__ == '__main__':
    _main()

####################################################################################################
