#!/usr/bin/env python3

""" Process the graph directory structure generating a static HTML gallery. """

import argparse
from collections import defaultdict
import cProfile
import io
import json
import logging
import os
from pprint import pformat, pprint
import pstats
import re
import sys

from jinja2 import Template

from deepdiff import DeepDiff
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

def _argument_parser():
    """ Argument parser for the stats parser. """
    parser = argparse.ArgumentParser(
        description='RLLIB & SUMO Gallery generator.')
    parser.add_argument(
        '--dir-tree', required=True, type=str, 
        help='Graphs directory.')
    parser.add_argument(
        '--exp', required=True, type=str, 
        help='Experiment name.')
    parser.add_argument(
        '--profiler', dest='profiler', action='store_true', help='Enable cProfile.')
    parser.set_defaults(profiler=False)
    return parser.parse_args()

def _main():
    """ Process the graph directory structure generating a static HTML gallery. """

    config = _argument_parser()

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler = cProfile.Profile()
        profiler.enable()
    ## ========================              PROFILER              ======================== ##

    gallery = HTMLGallery(config.dir_tree, config.exp)
    gallery.generate_aggregated()
    gallery.generate_agents()
    gallery.generate_policies()
    gallery.generate_qvalues()
    LOGGER.info('Done')

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler.disable()
        results = io.StringIO()
        pstats.Stats(profiler, stream=results).sort_stats('cumulative').print_stats(50)
        LOGGER.info('Profiler: \n%s', pformat(results.getvalue()))
    ## ========================              PROFILER              ======================== ##

class HTMLGallery():
    """ Generate a HTML Gallery. """

    REWARD_SUFFIX = ".aggregated-overview.png"
    AGENTS_SUFFIX = ".agents-decisions-overview.png"
    MODES_SUFFIX = ".mode-share-overview.png"

    AGGREGATED_TEMPLATE = """
<!DOCTYPE html>
<html>
    <head>
        <title>{{exp_name}}</title>
    </head>
    <style>
        img {
            border: 1px solid #ddd; /* Gray border */
            border-radius: 4px;  /* Rounded border */
            padding: 5px; /* Some padding */
            width: 300px; /* Set a small width */
        }

        /* Add a hover effect (blue shadow) */
        img:hover {
            box-shadow: 0 0 2px 1px rgba(0, 140, 186, 0.5);
        }
    </style>
    <body>
        <h1 style="text-align:center">{{exp_name}}</h1>

        <h2 style="text-align:center">Aggregated data</h2>
        <table style="margin-left:auto;margin-right:auto;">
        <tr>
            <td width="33%" style="text-align:center"><figure><a target="_blank" href="{{rewards}}"><img src="{{rewards}}"/></a><figcaption>Average reward</figcaption></figure></td>
            <td width="33%" style="text-align:center"><figure><a target="_blank" href="{{agents}}"><img src="{{agents}}"/></a><figcaption>Agents overview</figcaption></figure></td>
            <td width="33%" style="text-align:center"><figure><a target="_blank" href="{{modes}}"><img src="{{modes}}"/></a><figcaption>Modes overview</figcaption></figure></td>
        </tr>
        </table>
    </body>
</html>
    """

    AGENTS_TEMPLATE = """
<!DOCTYPE html>
<html>
    <head>
        <title>{{exp_name}}</title>
    </head>
    <style>
        img {
            border: 1px solid #ddd; /* Gray border */
            border-radius: 4px;  /* Rounded border */
            padding: 5px; /* Some padding */
            width: 300px; /* Set a small width */
        }

        /* Add a hover effect (blue shadow) */
        img:hover {
            box-shadow: 0 0 2px 1px rgba(0, 140, 186, 0.5);
        }
    </style>
    <body>
        <h1 style="text-align:center">{{exp_name}}</h1>

        <h2 style="text-align:center">Agents overview</h2>
        <table style="margin-left:auto;margin-right:auto;">
        {% for item in items %}
        <tr>
            <td width="25%" style="text-align:center"><figure><a target="_blank" href="{{item.img1}}"><img src="{{item.img1}}"/></a><figcaption>{{item.caption1}}</figcaption></figure></td>
            <td width="25%" style="text-align:center"><figure><a target="_blank" href="{{item.img2}}"><img src="{{item.img2}}"/></a><figcaption>{{item.caption2}}</figcaption></figure></td>
            <td width="25%" style="text-align:center"><figure><a target="_blank" href="{{item.img3}}"><img src="{{item.img3}}"/></a><figcaption>{{item.caption3}}</figcaption></figure></td>
            <td width="25%" style="text-align:center"><figure><a target="_blank" href="{{item.img4}}"><img src="{{item.img4}}"/></a><figcaption>{{item.caption4}}</figcaption></figure></td>
        </tr>
        {% endfor %}
        </table>
    </body>
</html>
    """

    POLICIES_TEMPLATE = """
<!DOCTYPE html>
<html>
    <head>
        <title>{{exp_name}}</title>
    </head>
    <style>
        img {
            border: 1px solid #ddd; /* Gray border */
            border-radius: 4px;  /* Rounded border */
            padding: 5px; /* Some padding */
            width: 300px; /* Set a small width */
        }

        /* Add a hover effect (blue shadow) */
        img:hover {
            box-shadow: 0 0 2px 1px rgba(0, 140, 186, 0.5);
        }
    </style>
    <body>
        <h1 style="text-align:center">{{exp_name}}</h1>

        <h2 style="text-align:center">Latest policy overview</h2>
        <table style="margin-left:auto;margin-right:auto;">
        {% for item in items %}
        <tr>
            <td width="25%" style="text-align:center"><figure><a target="_blank" href="{{item.img1}}"><img src="{{item.img1}}"/></a><figcaption>{{item.caption1}}</figcaption></figure></td>
            <td width="25%" style="text-align:center"><figure><a target="_blank" href="{{item.img2}}"><img src="{{item.img2}}"/></a><figcaption>{{item.caption2}}</figcaption></figure></td>
            <td width="25%" style="text-align:center"><figure><a target="_blank" href="{{item.img3}}"><img src="{{item.img3}}"/></a><figcaption>{{item.caption3}}</figcaption></figure></td>
            <td width="25%" style="text-align:center"><figure><a target="_blank" href="{{item.img4}}"><img src="{{item.img4}}"/></a><figcaption>{{item.caption4}}</figcaption></figure></td>
        </tr>
        {% endfor %}
        </table>
    </body>
</html>
    """

    QVALUES_TEMPLATE = """
<!DOCTYPE html>
<html>
    <head>
        <title>{{exp_name}}</title>
    </head>
    <style>
        img {
            border: 1px solid #ddd; /* Gray border */
            border-radius: 4px;  /* Rounded border */
            padding: 5px; /* Some padding */
            width: 300px; /* Set a small width */
        }

        /* Add a hover effect (blue shadow) */
        img:hover {
            box-shadow: 0 0 2px 1px rgba(0, 140, 186, 0.5);
        }
    </style>
    <body>
        <h1 style="text-align:center">{{exp_name}}</h1>

        <h2 style="text-align:center">Latest policy overview with q-values</h2>
        <table style="margin-left:auto;margin-right:auto;">
        {% for item in items %}
        <tr>
            <td width="25%" style="text-align:center"><figure><a target="_blank" href="{{item.img1}}"><img src="{{item.img1}}"/></a><figcaption>{{item.caption1}}</figcaption></figure></td>
            <td width="25%" style="text-align:center"><figure><a target="_blank" href="{{item.img2}}"><img src="{{item.img2}}"/></a><figcaption>{{item.caption2}}</figcaption></figure></td>
            <td width="25%" style="text-align:center"><figure><a target="_blank" href="{{item.img3}}"><img src="{{item.img3}}"/></a><figcaption>{{item.caption3}}</figcaption></figure></td>
            <td width="25%" style="text-align:center"><figure><a target="_blank" href="{{item.img4}}"><img src="{{item.img4}}"/></a><figcaption>{{item.caption4}}</figcaption></figure></td>
        </tr>
        {% endfor %}
        </table>
    </body>
</html>
"""

    def __init__(self, directory, exp):
        self.dir = directory
        self.experiment = exp

    @staticmethod
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    @staticmethod
    def get_item(lst, pos):
        return lst[pos] if pos < len(lst) else ''

    def generate_aggregated(self):
        """ Generates the gallery structure. """

        self.html_aggr_template = Template(self.AGGREGATED_TEMPLATE)
        self.html_aggr_string = None

        ## aggregated data
        rewards = None
        agents = None
        modes = None
        for item in os.listdir(os.path.join(self.dir, 'aggregated')):
            if self.REWARD_SUFFIX in item:
                rewards = os.path.join('aggregated', item)
            elif self.AGENTS_SUFFIX in item:
                agents = os.path.join('aggregated', item)
            elif self.MODES_SUFFIX in item:
                modes = os.path.join('aggregated', item)
        
        self.html_aggr_string = self.html_aggr_template.render(
            exp_name=self.experiment, rewards=rewards, agents=agents, modes=modes)

        with open(os.path.join(self.dir, 'aggregated.html'), 'w') as fstream:
            fstream.write(self.html_aggr_string)

    def generate_agents(self):
        """ Generates the gallery structure. """

        self.html_agents_template = Template(self.AGENTS_TEMPLATE)
        self.html_agents_string = None
        self.html_policies_template = Template(self.POLICIES_TEMPLATE)
        self.html_policies_string = None

        ## agents
        files = []
        for item in os.listdir(os.path.join(self.dir, 'agents')):
            if '.png' in item:
                files.append(item)
        agents_insight = []
        for chunk in self.chunks(files, 4):
            agents_insight.append({
                'img1': os.path.join('agents', self.get_item(chunk, 0)),
                'caption1': self.get_item(chunk, 0),
                'img2': os.path.join('agents', self.get_item(chunk, 1)),
                'caption2': self.get_item(chunk, 1),
                'img3': os.path.join('agents', self.get_item(chunk, 2)),
                'caption3': self.get_item(chunk, 2),
                'img4': os.path.join('agents', self.get_item(chunk, 3)),
                'caption4': self.get_item(chunk, 3),
            })

        self.html_agents_string = self.html_agents_template.render(
            exp_name=self.experiment, items=agents_insight)
        
        with open(os.path.join(self.dir, 'agents.html'), 'w') as fstream:
            fstream.write(self.html_agents_string)

    def generate_policies(self):
        """ Generates the gallery structure. """

        self.html_policies_template = Template(self.POLICIES_TEMPLATE)
        self.html_policies_string = None

        ## policies
        files = []
        for item in os.listdir(os.path.join(self.dir, 'policies')):
            if '.png' in item:
                files.append(item)
        policies = []
        for chunk in self.chunks(files, 4):
            policies.append({
                'img1': os.path.join('policies', self.get_item(chunk, 0)),
                'caption1': self.get_item(chunk, 0),
                'img2': os.path.join('policies', self.get_item(chunk, 1)),
                'caption2': self.get_item(chunk, 1),
                'img3': os.path.join('policies', self.get_item(chunk, 2)),
                'caption3': self.get_item(chunk, 2),
                'img4': os.path.join('policies', self.get_item(chunk, 3)),
                'caption4': self.get_item(chunk, 3),
            })

        self.html_policies_string = self.html_policies_template.render(
            exp_name=self.experiment, items=policies)

        with open(os.path.join(self.dir, 'policies.html'), 'w') as fstream:
            fstream.write(self.html_policies_string)

    def generate_qvalues(self):
        """ Generates the gallery structure. """

        self.html_qvalues_template = Template(self.QVALUES_TEMPLATE)
        self.html_qvalue_string = None

        ## qvalues
        files = []
        for item in os.listdir(os.path.join(self.dir, 'qvalues')):
            if '.png' in item:
                files.append(item)
        qvalues = []
        for chunk in self.chunks(files, 4):
            qvalues.append({
                'img1': os.path.join('qvalues', self.get_item(chunk, 0)),
                'caption1': self.get_item(chunk, 0),
                'img2': os.path.join('qvalues', self.get_item(chunk, 1)),
                'caption2': self.get_item(chunk, 1),
                'img3': os.path.join('qvalues', self.get_item(chunk, 2)),
                'caption3': self.get_item(chunk, 2),
                'img4': os.path.join('qvalues', self.get_item(chunk, 3)),
                'caption4': self.get_item(chunk, 3),
            })

        self.html_qvalues_string = self.html_qvalues_template.render(
            exp_name=self.experiment, items=qvalues)

        with open(os.path.join(self.dir, 'qvalues.html'), 'w') as fstream:
            fstream.write(self.html_qvalues_string)

####################################################################################################

if __name__ == '__main__':
    _main()

####################################################################################################