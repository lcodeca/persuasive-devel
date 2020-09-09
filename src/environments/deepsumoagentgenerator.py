#!/usr/bin/env python3

""" Generates JSON configuration file for a given scenario. """

import argparse
import collections
from copy import deepcopy
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import shapely.geometry as geometry

# """ Import SUMO library """
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
    import sumolib
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

####################################################################################################

def _argument_parser():
    """ Argument parser for the SUMOAgentGenerator. """
    parser = argparse.ArgumentParser(
        description='Generates JSON configuration file for a given scenario.')
    parser.add_argument('--net', required=True, type=str, help='The SUMO net.xml.')
    parser.add_argument('--shapes', required=True, type=str,
                        help='The SUMO shapefile with the TAZ shape.')
    parser.add_argument('--target', required=True, type=str,
                        help='The ID of the target area in the SUMO shapefile.')
    parser.add_argument('--num', default=10, type=int, help='Number of agent to be generated.')
    parser.add_argument('--out', default='output.agents.josm', type=str,
                        help='The JOSM output file.')
    parser.add_argument('--default', required=True, type=str,
                        help='The default agent settings (JOSM file).')
    parser.add_argument('--origin', nargs=2, type=float, default=[],
                        help='The default x, y for the origin.')
    parser.add_argument('--destination', nargs=2, type=float, default=[],
                        help='The default x, y for the destination.')
    parser.add_argument('--random-seed', default=0, type=int, dest='seed',
                        help='If set, randomize a seed from 0 to the passed integer.')
    parser.add_argument('--random-start', nargs=2, type=int, default=[], dest='start',
                        help='If set, randomize the start using the passed interval.')
    return parser.parse_args()

####################################################################################################

class SUMOAgentGenerator(object):
    """ Generates JSON configuration file for a given scenario. """

    def __init__(self, config):
        """ config.net          --> 'The SUMO net.xml.'
            config.shapes       --> 'The SUMO shapefile with the TAZ shape.'
            config.target       --> 'The ID of the target area in the SUMO shapefile.'
            config.num          --> 'Number of agent to be generated.'
            config.out          --> 'The JOSM output file.'
            config.default      --> 'The default agent settings (JOSM).'
            config.origin       --> 'The default x, y for the origin.'
            config.destination  --> 'The default x, y for the destination.'
            config.seed         --> 'If set, randomize a seed from 0 to the passed integer.
            config.start        --> 'If set, randomize the start using the passed interval.'
        """
        self._config = config
        self._sumo_net = sumolib.net.readNet(self._config.net)
        self._edges = list()
        self._load_edges()
        self._shapes = dict()
        self._load_shapefile()
        self._agents = collections.defaultdict(dict)
        self._all_starts = list()
        self._all_origins = list()
        self._all_destinations = list()
        self._generate_agents()
        self._save_to_file()
        self._plot_departure()
        self._density_plot(self._all_origins, 'origins')
        self._density_plot(self._all_destinations, 'destinations')

    ################################################################################################

    def _load_shapefile(self):
        """ Load SUMO net.xml and extract the edges """
        self._sumo_shapes = sumolib.shapes.polygon.read(self._config.shapes, includeTaz=True)
        for poly in self._sumo_shapes:
            self._shapes[poly.id] = {
                'shape': poly.shape,
                'bounding_box': poly.getBoundingBox(),
                'convex_hull': geometry.MultiPoint(poly.shape).convex_hull
            }

    def _load_edges(self):
        """ Load SUMO net.xml and extract the edges """
        for edge in self._sumo_net.getEdges():
            if not edge.allows('pedestrian'):
                continue
            name = edge.getID()
            if ':' not in name:
                self._edges.append(name)

    def _generate_random_coords_in_area(self):
        """
            Generates the random coords using the bounding box and fit them in the shape.
            Example of bounding box: (-146.63, -339.13, 4043.48, 3838.63)
        """
        min_x, min_y, max_x, max_y = self._shapes[self._config.target]['bounding_box']
        convex_hull = self._shapes[self._config.target]['convex_hull']
        x_coord = np.random.uniform(min_x, max_x)
        y_coord = np.random.uniform(min_y, max_y)
        while not convex_hull.contains(geometry.Point(x_coord, y_coord)):
            x_coord = np.random.uniform(min_x, max_x)
            y_coord = np.random.uniform(min_y, max_y)
        return [x_coord, y_coord]


    def _generate_agents(self):
        """ Generate the agents """
        default = json.load(open(self._config.default))
        for agent in tqdm(range(self._config.num)):
            agent = 'agent_{}'.format(agent)
            if self._config.seed > 0:
                default['seed'] = np.random.randint(0, self._config.seed + 1)
            if self._config.start:
                begin, end = self._config.start
                default['start'] = np.random.randint(begin, end + 1)
            if self._config.origin:
                default['origin'] = self._config.origin
            else:
                default['origin'] = self._generate_random_coords_in_area()
            if self._config.destination:
                default['destination'] = self._config.destination
            else:
                default['destination'] = self._generate_random_coords_in_area()
            self._agents[agent] = deepcopy(default)
            self._all_starts.append(self._agents[agent]['start'])
            self._all_origins.append(self._agents[agent]['origin'])
            self._all_destinations.append(self._agents[agent]['destination'])

    ################################################################################################

    def _save_to_file(self):
        """ Save the agents in a JSON file. """
        json.dump(self._agents, open(self._config.out, 'w'))

    def _plot_departure(self):
        fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(15, 10))
        axes.hist(np.array(self._all_starts)/3600, 60, density=True, facecolor='g', alpha=0.75)
        axes.set_title('Starting Time [h]')
        axes.grid(True)
        plt.show()
        fig.savefig('{}.starts.svg'.format(self._config.out),
                    dpi=300, transparent=False, bbox_inches='tight')

    def _density_plot(self, points, tag):
        convex_hull = self._shapes[self._config.target]['convex_hull']
        x, y = np.array(points).T

        fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
        axes.set_title('Scatterplot of {}'.format(tag))
        axes.plot(x, y, 'ko', color='g', alpha=0.75)
        axes.plot(*convex_hull.exterior.xy)
        axes.set_aspect('equal', 'box')
        axes.grid()
        plt.show()
        fig.savefig('{}.scatter.{}.svg'.format(self._config.out, tag),
                    dpi=300, transparent=False, bbox_inches='tight')

        fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
        nbins = 50
        axes.set_title('Hexbin of {}'.format(tag))
        axes.plot(*convex_hull.exterior.xy)
        hb = axes.hexbin(x, y, gridsize=nbins, cmap=plt.cm.BuGn)
        cb = fig.colorbar(hb, ax=axes)
        cb.set_label('counts')
        axes.set_aspect('equal', 'box')
        axes.legend()
        plt.show()
        fig.savefig('{}.hexbin.{}.svg'.format(self._config.out, tag),
                    dpi=300, transparent=False, bbox_inches='tight')
        # matplotlib.pyplot.close('all')

    ################################################################################################

####################################################################################################

if __name__ == '__main__':

    cmd_args = _argument_parser()
    _ = SUMOAgentGenerator(cmd_args)
