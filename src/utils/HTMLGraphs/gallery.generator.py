#!/usr/bin/env python3

""" Process the graph directory structure generating a static HTML gallery. """

import argparse
import cProfile
import io
import logging
import os
from pprint import pformat
import pstats
import re

from jinja2 import Template

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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
    try:
        gallery.generate_aggregated()
    except FileNotFoundError as err:
        logger.error("Impossible to generate gallery: '%s'", str(err))

    try:
        gallery.generate_agents()
    except FileNotFoundError as err:
        logger.error("Impossible to generate gallery: '%s'", str(err))

    try:
        gallery.generate_policies()
    except FileNotFoundError as err:
        logger.error("Impossible to generate gallery: '%s'", str(err))

    try:
        gallery.generate_qvalues()
    except FileNotFoundError as err:
        logger.error("Impossible to generate gallery: '%s'", str(err))
    try:
        gallery.generate_qvalues_evol()
    except FileNotFoundError as err:
        logger.error("Impossible to generate gallery: '%s'", str(err))

    logger.info('Done')

    ## ========================              PROFILER              ======================== ##
    if config.profiler:
        profiler.disable()
        results = io.StringIO()
        pstats.Stats(profiler, stream=results).sort_stats('cumulative').print_stats(50)
        logger.info('Profiler: \n%s', pformat(results.getvalue()))
    ## ========================              PROFILER              ======================== ##

class HTMLGallery():
    """ Generate a HTML Gallery. """

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

    @staticmethod
    def alphanumeric_sort(iterable):
        """
        Sorts the given iterable in the way that is expected.
        See: https://stackoverflow.com/questions/2669059/how-to-sort-alpha-numeric-set-in-python
        """
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(iterable, key=alphanum_key)

    ################################################################################################

    REWARD_SUFFIX = "aggregated-overview.png"
    AGENTS_SUFFIX = "agents-decisions-overview.png"
    MODES_SUFFIX = "mode-share-overview.png"

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
</html>"""

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
                # print('Found: {} --> {}'.format(item, rewards))
            elif self.AGENTS_SUFFIX in item:
                agents = os.path.join('aggregated', item)
                # print('Found: {} --> {}'.format(item, agents))
            elif self.MODES_SUFFIX in item:
                modes = os.path.join('aggregated', item)
                # print('Found: {} --> {}'.format(item, modes))

        self.html_aggr_string = self.html_aggr_template.render(
            exp_name=self.experiment, rewards=rewards, agents=agents, modes=modes)

        with open(os.path.join(self.dir, 'aggregated.html'), 'w') as fstream:
            fstream.write(self.html_aggr_string)

    ################################################################################################

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
</html>"""

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

    ################################################################################################

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
</html>"""

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

    ################################################################################################

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
</html>"""

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

    ################################################################################################

    QVALUES_EVOL_TEMPLATE = """
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
            width: 800px; /* Set a small width */
        }

        /* Add a hover effect (blue shadow) */
        img:hover {
            box-shadow: 0 0 2px 1px rgba(0, 140, 186, 0.5);
        }
    </style>
    <body>
        <h1 style="text-align:center">{{exp_name}}</h1>
        <h2 style="text-align:center">{{agent}}</h2>
        <h2 style="text-align:center">Policy overview: Q-values evolution</h2>
        <table style="margin-left:auto;margin-right:auto;">
        {% for item in items %}
        <tr>
            <td width="100%" style="text-align:center">
                <figure>
                    <a target="_blank" href="{{item.img}}">
                        <img src="{{item.img}}"/>
                    </a>
                    <figcaption>{{item.caption}}</figcaption>
                </figure>
            </td>
        </tr>
        {% endfor %}
        </table>
    </body>
</html>"""

    def generate_qvalues_evol(self):
        """ Generates the gallery structure. """

        self.html_qvalues_evol_template = Template(self.QVALUES_EVOL_TEMPLATE)
        self.html_qvalue_evol_string = None

        ## qvalues
        for agent_dir in os.listdir(os.path.join(self.dir, 'qvalues-evol')):
            files = []
            for item in os.listdir(os.path.join(self.dir, 'qvalues-evol', agent_dir)):
                if '.png' in item:
                    files.append(item)
            qvalues = []
            for fname in self.alphanumeric_sort(files):
                qvalues.append({
                    'img': os.path.join('qvalues-evol', agent_dir, fname),
                    'caption': fname,
                })

            self.html_qvalues_evol_string = self.html_qvalues_evol_template.render(
                exp_name=self.experiment, agent=agent_dir, items=qvalues)

            gallery_fname = 'qvalues-{}-evol.html'.format(agent_dir)
            with open(os.path.join(self.dir, gallery_fname), 'w') as fstream:
                fstream.write(self.html_qvalues_evol_string)

####################################################################################################

if __name__ == '__main__':
    _main()

####################################################################################################
