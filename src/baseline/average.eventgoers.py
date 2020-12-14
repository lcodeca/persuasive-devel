#!/usr/bin/env python3

"""
    Implementation of the BASELINE for Persuasive, based on an average person taking decision
    based on an app such as Google Traffic.
"""

import argparse
import collections
import csv
import json
import math
import os
import sys
import traceback
import statistics

from datetime import datetime
from decimal import Decimal
from pprint import pprint, pformat

import numpy as np
from numpy.random import RandomState
import shapely.geometry as geometry

# """ Import SUMO library """
if 'SUMO_TOOLS' in os.environ:
    sys.path.append(os.environ['SUMO_TOOLS'])
    import traci
    import sumolib
    from traci.exceptions import TraCIException
    import traci.constants as tc
else:
    sys.exit("Please declare environment variable 'SUMO_TOOLS'")

####################################################################################################

def argument_parser():
    """ Argument parser for the trainer"""
    parser = argparse.ArgumentParser(description='BASELINE for Persuasive.')
    parser.add_argument('--sumo-cfg', required=True, type=str, help='SUMO config file.')
    parser.add_argument('--agents-cfg', required=True, type=str, help='Agents config file.')
    parser.add_argument('--env-cfg', required=True, type=str, help='Environment config file.')
    parser.add_argument('--output', required=True, type=str, help='Output directory.')
    parser.add_argument('--profiler', action='store_true', help='Enables cProfile.')
    parser.set_defaults(profiler=False)
    return parser.parse_args()

ARGS = argument_parser()

####################################################################################################

class Agent():
    def __init__(self, origin, destination, start_time, expected_arrival):
        self.origin = origin
        self.destination = destination
        self.start_time = start_time
        self.expected_arrival = expected_arrival

        # Additionals
        self.mode = None
        self.departure = float('NaN')
        self.arrival = float('NaN')

        # Metrics
        self.travel = float('NaN')
        self.waiting = float('NaN')
        self.lateness = float('NaN')

    def finish(self):
        # print(self.departure, self.arrival)
        self.travel = self.arrival - self.departure
        if self.arrival <= self.expected_arrival:
            self.waiting = self.expected_arrival - self.arrival
        else:
            self.lateness = self.arrival - self.expected_arrival
        # print(self.travel, self.waiting, self.lateness)

    def to_dict(self):
        return {
            'origin': self.origin,
            'destination': self.destination,
            'start_time': self.start_time,
            'expected_arrival': self.expected_arrival,
            'mode': self.mode,
            'departure': self.departure,
            'arrival': self.arrival,
            'travel': self.travel,
            'waiting': self.waiting,
            'lateness': self.lateness,
        }

    def __str__(self):
        return '{} - {} - {} : {} - {} - {} : {} - {} - {} : {}'.format(
            self.origin, self.destination, self.expected_arrival,
            self.start_time, self.departure, self.arrival,
            self.travel, self.waiting, self.lateness,
            self.mode)

class SimEventGoers():
    """
    Eventgoers based on an average person taking decision based on an app such as Google Traffic.
    """

    modes_w_vehicles = ['passenger', 'bicycle', 'ptw', 'on-demand',]

    def __init__(self, agents, environment, sumo_cfg):

        ## Load the config files for the agents and the environmet
        self.agents_init = json.load(open(ARGS.agents_cfg))
        self.env_init = json.load(open(ARGS.env_cfg))

        # Random number generator
        self.rndgen = RandomState() # self.env_init['seed']

        ## SUMO network
        self.network = sumolib.net.readNet(self.env_init['sumo_net_file'])

        ## Start simulation
        traci.start(['sumo', '-c', ARGS.sumo_cfg])

        ## Create Agents
        self._create_agents()

        ## Sort Agents
        self.waiting_agents = list()
        for agent_id, vals in self.agents.items():
            self.waiting_agents.append((vals.start_time, agent_id))
        self.waiting_agents.sort()

        self.ready_agents = []

        self.active = set()

        # aggregated metrics
        self.departures = []
        self.arrivals = []
        self.travel_times = []
        self.waitings = []
        self.latenesses = []
        self.modes = collections.defaultdict(int)

    def simulate(self):
        while (traci.simulation.getMinExpectedNumber() > 0 and
               (self.waiting_agents or self.ready_agents or self.active)):
            traci.simulationStep()
            now = traci.simulation.getTime()
            # print('Waiting:', len(self.waiting_agents))
            # print('Ready:', len(self.ready_agents))

            # SAVE THE METRICS FROM THE PREVIOUS SIM STEP
            subs = traci.person.getAllSubscriptionResults()
            current = set(subs.keys())
            # print('Current:', current, len(current))
            departed = current - self.active
            # print('Departed:', departed, len(departed))
            for agent_id in departed:
                self.agents[int(agent_id)].departure = now
            arrived = self.active - current
            # print('Arrived:', arrived, len(arrived))
            for agent_id in arrived:
                self.agents[int(agent_id)].arrival = now
                self.agents[int(agent_id)].finish()
                print('[{}] DONE - Agent {}: {}'.format(
                    now, agent_id, str(self.agents[int(agent_id)])))
                self._save_metrics(int(agent_id))
            self.active = current
            # print('Active:', self.active, len(self.active))

            # WAKE UP THE AGENTS
            while self.waiting_agents and self.waiting_agents[0][0] <= now:
                start_time, agent_id = self.waiting_agents.pop(0)
                departure, route = self._google_traffic_user(agent_id, now)
                self.ready_agents.append((departure, (agent_id, route)))
                # print('[{}] Awakening agent {} at time {} and schedule it at {}'.format(
                #     now, agent_id, start_time, departure))

            #  INSERT THE AGENTS
            self.ready_agents.sort()
            while self.ready_agents and self.ready_agents[0][0] < now:
                departure, (agent_id, route) = self.ready_agents.pop(0)
                self._insert_agent_in_sim(agent_id, route)
                # print('[{}] Inserting agent {} at time {}'.format(
                #     now, agent_id, departure))

        ########################################################################

    def _save_metrics(self, agent_id):
        self.departures.append(self.agents[agent_id].departure)
        self.arrivals.append(self.agents[agent_id].arrival)
        self.travel_times.append(self.agents[agent_id].travel)
        self.waitings.append(self.agents[agent_id].waiting)
        self.latenesses.append(self.agents[agent_id].lateness)
        self.modes[self.agents[agent_id].mode] += 1

    def _save_json(self, location, tag):
        report = {}
        report['global'] = self.agents_init
        for agent_id, agent in self.agents.items():
            report[agent_id] = agent.to_dict()
        fname = os.path.join(location, '[{}]report.json'.format(tag))
        with open(fname, 'w') as jfile:
            json.dump(report, jfile, indent=2)
        print('Saved:', fname)

    def _save_cumulative_csv(self, location, tag):
        report = {}
        report['id'] = tag
        report['departure_mean'] = np.nanmean(self.departures)
        report['departure_median'] = np.nanmedian(self.departures)
        report['departure_std'] = np.nanstd(self.departures)
        report['departure_min'] = np.nanmin(self.departures)
        report['departure_max'] = np.nanmax(self.departures)
        report['arrivals_mean'] = np.nanmean(self.arrivals)
        report['arrivals_median'] = np.nanmedian(self.arrivals)
        report['arrivals_std'] = np.nanstd(self.arrivals)
        report['arrivals_min'] = np.nanmin(self.arrivals)
        report['arrivals_max'] = np.nanmax(self.arrivals)
        report['travel_times_mean'] = np.nanmean(self.travel_times)
        report['travel_times_median'] = np.nanmedian(self.travel_times)
        report['travel_times_std'] = np.nanstd(self.travel_times)
        report['travel_times_min'] = np.nanmin(self.travel_times)
        report['travel_times_max'] = np.nanmax(self.travel_times)
        report['waitings_mean'] = np.nanmean(self.waitings)
        report['waitings_median'] = np.nanmedian(self.waitings)
        report['waitings_std'] = np.nanstd(self.waitings)
        report['waitings_min'] = np.nanmin(self.waitings)
        report['waitings_max'] = np.nanmax(self.waitings)
        report['latenesses_mean'] = np.nanmean(self.latenesses)
        report['latenesses_median'] = np.nanmedian(self.latenesses)
        report['latenesses_std'] = np.nanstd(self.latenesses)
        report['latenesses_min'] = np.nanmin(self.latenesses)
        report['latenesses_max'] = np.nanmax(self.latenesses)
        pprint(report)
        for mode in self.agents_init['modes']:
            report[mode] = self.modes[mode]
        fname = os.path.join(location, 'cumulative_report.csv')
        if os.path.isfile(fname):
            with open(fname, 'a') as f:
                w = csv.DictWriter(f, sorted(report.keys()))
                w.writerow(report)
        else: # Create with the header
            with open(fname, 'w') as f:
                w = csv.DictWriter(f, sorted(report.keys()))
                w.writeheader()
                w.writerow(report)
        print('Saved:', fname)

    def save_results(self, location):
        tag = datetime.now().strftime("%d-%b-%Y(%H:%M:%S.%f)")
        self._save_cumulative_csv(location, tag)
        self._save_json(location, tag)

    ################# HEURISTIC

    @staticmethod
    def get_mode_parameters(mode):
        """
        Return the correst TraCI parameters for the requested mode.
        See: https://sumo.dlr.de/docs/TraCI/Simulation_Value_Retrieval.html
                    #command_0x87_find_intermodal_route
        Param: mode, String.
        Returns: _mode, _ptype, _vtype
        """
        if mode == 'public':
            return 'public', '', ''
        if mode == 'bicycle':
            return 'bicycle', '', 'bicycle'
        if mode == 'walk':
            return '', 'pedestrian', ''
        return 'car', '', mode      # (but car is not always necessary, and it may
                                    #  creates unusable alternatives)

    def is_valid_route(self, mode, route):
        """
        Handle findRoute and findIntermodalRoute results.

        Params:
            mode, String.
            route, return value of findRoute or findIntermodalRoute.
        """
        if route is None:
            # traci failed
            return False
        _mode, _ptype, _vtype = self.get_mode_parameters(mode)
        if not isinstance(route, (list, tuple)):
            # only for findRoute
            if len(route.edges) >= 2:
                return True
        elif _mode == 'public':
            for stage in route:
                if stage.line:
                    return True
        elif _mode in ('car', 'bicycle'):
            for stage in route:
                if stage.type == tc.STAGE_DRIVING and len(stage.edges) >= 2:
                    return True
        else:
            for stage in route:
                if len(stage.edges) >= 2:
                    return True
        return False

    def _google_traffic_user(self, agent_id, now):
        plans = {}
        for mode in self.agents_init['modes']:
            # compute the route using findIntermodalRoute
            _mode, _ptype, _vtype = self.get_mode_parameters(mode)
            try:
                route = traci.simulation.findIntermodalRoute(
                    self.agents[agent_id].origin, self.agents[agent_id].destination,
                    modes=_mode, pType=_ptype, vType=_vtype, routingMode=1)
                if not self.is_valid_route(mode, route):
                    route = None
            except traci.exceptions.TraCIException:
                route = None
            if route:
                # print('Selected mode: {}. [mode "{}", ptype "{}", vtype "{}"]'.format(
                #     mode, _mode, _ptype, _vtype))
                # print('Route: {}'.format(route))
                walking_ett = 0.0
                for stage in route:
                    if stage.type == tc.STAGE_WALKING:
                        walking_ett += stage.travelTime
                # print(mode, walking_ett)
                plans[mode] = (walking_ett, route)

        ## Remove solutions that require too much walking:
        usable_plans = []
        usable_modes = []
        min_alternative_mode = None
        min_alternative_route = None
        min_alternative_w_ett = None
        for mode, (walking_ett, route) in plans.items():
            if walking_ett < self.agents_init['max-walking-min'] * 60.0:
                usable_modes.append(mode)
                usable_plans.append(route)
            else:
                if min_alternative_w_ett is None:
                    min_alternative_mode = mode
                    min_alternative_route = route
                    min_alternative_w_ett = walking_ett
                elif min_alternative_w_ett > walking_ett:
                    min_alternative_mode = mode
                    min_alternative_route = route
                    min_alternative_w_ett = walking_ett

        ## Decision making time!
        choice = None
        if not usable_plans:
            ## FALL BACK TO MIN BETWEEN THE AVAILABLE
            choice = min_alternative_route
            self.agents[agent_id].mode = min_alternative_mode
            print('ALERT: using fallback mode', mode, walking_ett)
        else:
            probabilities = self._masked_probability(usable_modes)
            choice = self.rndgen.choice(np.arange(len(usable_plans)), p=probabilities)
            self.agents[agent_id].mode = usable_modes[choice]
            choice = usable_plans[choice]

        ett = 0.0
        for stage in choice:
            ett += stage.travelTime

        departure = self.agents_init['expected-arrival-time'] - \
            self.rndgen.uniform(0, self.agents_init['arrival-buffer-min']) * 60 - ett

        return departure, choice

    def _masked_probability(self, modes):
        total = 0.0
        for mode in modes:
            total += self.agents_init['modes'][mode]
            # print(mode, self.agents_init['modes'][mode], total)
        probabilities = []
        for mode in modes:
            probabilities.append(self.agents_init['modes'][mode] / total)
            # print(self.agents_init['modes'][mode], total, self.agents_init['modes'][mode] * total)
        if round(math.fsum(probabilities), 1) != 1.0:
            raise Exception(
                modes, self.agents_init['modes'], total, probabilities,
                round(math.fsum(probabilities), 1))
        return probabilities

    ################# AGENT INSERTION

    def _insert_agent_in_sim(self, agent_id, route):
        """ Insert an agent in the simulation, based on the given route. """
        # print('Adding person {}'.format(agent_id))
        traci.person.add(str(agent_id), self.agents[agent_id].origin, 0.0)
        veh_counter = 0
        # print('Processing stages...')
        for stage in route:
            # pprint(stage)
            if stage.type == tc.STAGE_DRIVING and stage.vType in self.modes_w_vehicles:
                vehicle_name = '{}_{}_tr'.format(agent_id, veh_counter)
                route_name = '{}_rou'.format(vehicle_name)
                traci.route.add(route_name, stage.edges)
                traci.vehicle.add(vehicle_name, route_name, typeID=stage.vType, depart='triggered')
                stage.line = vehicle_name
                veh_counter += 1
            traci.person.appendStage(str(agent_id), stage)
        # print('Adding person {} to the subscriptions'.format(agent_id))
        traci.person.subscribe(str(agent_id), (tc.VAR_ROAD_ID, tc.VAR_LANEPOSITION))

    ################# AGENT GENERATION

    def _create_agents(self):
        """ Main creation function. """
        self.bounding_box = {
            'bottom_left_X': self.env_init['bounding_box'][0],
            'bottom_left_Y': self.env_init['bounding_box'][1],
            'top_right_X': self.env_init['bounding_box'][2],
            'top_right_Y': self.env_init['bounding_box'][3],
        }
        # Convex hull initialization
        self.taz_shape = geometry.MultiPoint(self.agents_init['taz-shape']).convex_hull

        self.uniform_start_t = self.rndgen.uniform(
            self.agents_init['uniform-start'][0], self.agents_init['uniform-start'][1],
            self.agents_init['agents'])

        self.agents = dict()
        for agent_id in range(self.agents_init['agents']):
            self._create_agent(agent_id)

    def _generate_random_coords_in_area(self):
        """ Generates the random coords using the bounding box and fit them in the shape. """
        x_coord = self.rndgen.uniform(
            self.bounding_box['bottom_left_X'], self.bounding_box['top_right_X'])
        y_coord = self.rndgen.uniform(
            self.bounding_box['bottom_left_Y'], self.bounding_box['top_right_Y'])
        while not self.taz_shape.contains(geometry.Point(x_coord, y_coord)):
            x_coord = self.rndgen.uniform(
                self.bounding_box['bottom_left_X'], self.bounding_box['top_right_X'])
            y_coord = self.rndgen.uniform(
                self.bounding_box['bottom_left_Y'], self.bounding_box['top_right_Y'])
        return [x_coord, y_coord]

    def _create_agent(self, agent_id):
        """ Create a single agent. """

        ## DESTINATION
        destination_x, destination_y = self.agents_init['destination']
        edges = self.network.getNeighboringEdges(
            destination_x, destination_y, r=1000, includeJunctions=False, allowFallback=True)
        destination = None
        for distance, edge in sorted([(dist, edge) for edge, dist in edges]):
            if edge.allows('pedestrian'):
                destination = edge.getID()
                if distance > 500:
                    print('[{}] Destination {}, {} is {} from edge {}'.format(
                        agent_id, destination_x, destination_y, distance, destination))
                break
        if destination is None:
            raise Exception('Destination not foud for agent {}'.format(agent_id))

        ## ORIGIN
        origin_x, origin_y = self._generate_random_coords_in_area()
        edges = self.network.getNeighboringEdges(
            origin_x, origin_y, r=1000, includeJunctions=False, allowFallback=True)
        origin = None
        for distance, edge in sorted([(dist, edge) for edge, dist in edges]):
            if edge.allows('pedestrian'):
                origin = edge.getID()
                if origin == destination:
                    continue
                if distance > 500:
                    print('[{}] Origin {}, {} is {} from edge {}'.format(
                        agent_id, origin_x, origin_y, distance, origin))
                break
        if origin is None:
            raise Exception('Origin not foud for agent {}'.format(agent_id))

        # Starting time distribution: uniform

        self.agents[agent_id] = Agent(
            origin, destination, self.uniform_start_t[agent_id],
            self.agents_init['expected-arrival-time'])

    ############################################################################

def _main():
    """ Example of integration of triggers with PyPML. """
    sim = SimEventGoers(ARGS.agents_cfg, ARGS.env_cfg, ARGS.sumo_cfg)
    sim.simulate()
    sim.save_results(ARGS.output)

if __name__ == '__main__':
    ## ========================              PROFILER              ======================== ##
    # import cProfile, pstats, io
    # profiler = cProfile.Profile()
    # profiler.enable()
    ## ========================              PROFILER              ======================== ##
    try:
        _main()
    except traci.exceptions.TraCIException:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=10, file=sys.stdout)
    finally:
        traci.close()
        ## ========================              PROFILER              ======================== ##
        # profiler.disable()
        # results = io.StringIO()
        # pstats.Stats(profiler, stream=results).sort_stats('cumulative').print_stats(25)
        # print(results.getvalue())
        ## ========================              PROFILER              ======================== ##

####################################################################################################
