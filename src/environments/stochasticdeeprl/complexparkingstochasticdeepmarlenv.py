#!/usr/bin/env python3

"""
Stochastic MARL Environment based on ComplexStochasticPersuasiveDeepMARLEnv reward where the
passenger & ptw actions require finding a parking space.
"""

import collections
from enum import Enum
import logging
import os
import sys

from collections import defaultdict
from copy import deepcopy
from pprint import pprint, pformat

import numpy as np
import shapely.geometry as geometry
import xml.etree.ElementTree

import gym
import ray

from environments.stochasticdeeprl.complexstochasticdeepmarlenv import \
    ComplexDeepSUMOAgents, ComplexStochasticPersuasiveDeepMARLEnv

from utils.logger import set_logging

# """ Import SUMO library """
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
    import traci
    import libsumo
    import traci.constants as tc
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

####################################################################################################

DEBUGGER = True
logger = set_logging(__name__)

####################################################################################################

def env_creator(config):
    """ Environment creator used in the environment registration. """
    logger.debug('[env_creator] Environment creation: CSParkingPersuasiveDeepMARLEnv')
    return CSParkingPersuasiveDeepMARLEnv(config)

####################################################################################################

class ComplexParkingDeepSUMOAgents(ComplexDeepSUMOAgents):
    """ SUMO agent that computes specific max penalties given origin and destination. """

    def __init__(self, config, network, edge_to_parking):
        """ Initialize the environment. """
        super().__init__(config, network)
        self._edge_to_parking = edge_to_parking
        self._parking_cache = dict()

    def step(self, action, handler):
        """ Implements the logic of each specific action passed as input. """

        if self.inserted:
            logger.error('Agent %s has already been inserted in the simulation. [%s]',
                         self.agent_id, self.__repr__())
            return self.OUTCOME.ERROR # This cannot happen if the ENV is isolated.

        mode = None

        if action == 0:
            ### WAIT, nothing to do here.
            self.waited_steps += 1
            return self.OUTCOME.WAITING
        elif action in self.action_to_mode:
            self.inserted = True
            mode = self.action_to_mode[action]
        else:
            raise NotImplementedError('Action {} is not implemented.'.format(action))

        self.chosen_mode = mode

        _mode, _ptype, _vtype = handler.get_mode_parameters(mode)
        logger.debug('Selected mode: %s. [mode %s, ptype %s, vtype %s]',
                     mode, _mode, _ptype, _vtype)


        if mode in ['ptw', 'passenger']:
            # TRIP WITH PARKING REQUIREMENTS
            p_id, p_edge, _last_mile = self.find_closest_parking(
                self.destination, handler.traci_handler)
            if _last_mile:
                # we managed to find a parking close to the destination
                try:
                    route = handler.traci_handler.simulation.findIntermodalRoute(
                        self.origin, p_edge, modes=_mode, pType=_ptype, vType=_vtype, routingMode=1)
                    if route and not isinstance(route, list):
                        route = list(route)
                    if handler.is_valid_route(mode, route):
                        route[-1].destStop = p_id
                        route.extend(_last_mile)
                    else:
                        route = None
                except (traci.exceptions.TraCIException, libsumo.libsumo.TraCIException):
                    logger.warning(
                        'Parking failure for agent %s [%s]', str(self.agent_id), str(mode))
                    route = None
        else:
            # compute the route using findIntermodalRoute
            try:
                route = handler.traci_handler.simulation.findIntermodalRoute(
                    self.origin, self.destination, arrivalPos=100.0,
                    modes=_mode, pType=_ptype, vType=_vtype, routingMode=1)
                if not handler.is_valid_route(mode, route):
                    route = None
            except (traci.exceptions.TraCIException, libsumo.libsumo.TraCIException):
                route = None

        # generate the person trip
        if route:
            try:
                logger.debug('Adding person %s', str(self.agent_id))
                # add(self, personID, edgeID, pos, depart=-3, typeID='DEFAULT_PEDTYPE')
                handler.traci_handler.person.add(self.agent_id, self.origin, 0.0)
                veh_counter = 0
                for stage in route:
                    self.cost += stage.cost
                    self.ett += stage.travelTime
                    # appendStage(self, personID, stage)
                    if DEBUGGER:
                        logger.debug('%s', pformat(stage))
                    if stage.type == tc.STAGE_DRIVING and stage.vType in self.modes_w_vehicles:
                        vehicle_name = '{}_{}_tr'.format(self.agent_id, veh_counter)
                        route_name = '{}_rou'.format(vehicle_name)
                        # route.add(self, routeID, edges)
                        handler.traci_handler.route.add(route_name, stage.edges)
                        # vehicle.add(self, vehID, routeID, typeID='DEFAULT_VEHTYPE', depart=None,
                        #       departLane='first', departPos='base', departSpeed='0',
                        #       arrivalLane='current', arrivalPos='max', arrivalSpeed='current',
                        #       fromTaz='', toTaz='', line='', personCapacity=0, personNumber=0)
                        handler.traci_handler.vehicle.add(
                            vehicle_name, route_name, typeID=stage.vType, depart='triggered')
                        if mode in ['ptw', 'passenger']:
                            flags = ( 1 * 1 + # parking +
                                      2 * 1 + # personTriggered +
                                      4 * 0 + # containerTriggered +
                                      8 * 0 + # isBusStop +
                                     16 * 0 + # isContainerStop +
                                     32 * 0 + # chargingStation +
                                     64 * 1 ) # parkingarea
                            handler.traci_handler.vehicle.setStop(
                                vehicle_name, stage.destStop, flags=flags)
                        stage.line = vehicle_name
                        veh_counter += 1
                    handler.traci_handler.person.appendStage(self.agent_id, stage)
                handler.traci_handler.person.subscribe(self.agent_id,
                                                    (tc.VAR_ROAD_ID, tc.VAR_LANEPOSITION))
                return self.OUTCOME.INSERTED
            except (traci.exceptions.TraCIException, libsumo.libsumo.TraCIException) as exception:
                error = str(exception)
                if 'already exists' in error:
                    raise Exception(os.getpid(), error)
                self.chosen_mode = None
                self.chosen_mode_error = 'TraCIException for mode {}'.format(mode)
                self.cost = float('NaN')
                self.ett = float('NaN')
                logger.warning('Route not usable for %s using mode %s [%s]',
                            self.agent_id, mode, error)
                return self.OUTCOME.WRONG_DECISION # wrong decision, paid badly at the end

        self.chosen_mode = None
        self.chosen_mode_error = 'Invalid route using mode {}'.format(mode)
        self.cost = float('NaN')
        self.ett = float('NaN')
        logger.debug('Route not found for %s using mode %s', self.agent_id, mode)
        return self.OUTCOME.WRONG_DECISION # wrong decision, paid badly at the end

    ############################################################################

    def find_closest_parking(self, edge, sumo_handler):
        """ Given and edge, find the closest parking area. """
        cost = sys.float_info.max

        ret = self._check_parkings_cache(edge)
        if ret:
            return ret

        p_id = None

        for p_edge, parkings in self._edge_to_parking.items():
            p_id = np.random.choice(parkings)
            try:
                route = sumo_handler.simulation.findIntermodalRoute(
                    p_edge, edge, arrivalPos=100.0, pType="pedestrian")
            except (traci.exceptions.TraCIException, libsumo.libsumo.TraCIException):
                route = None
            if route:
                _cost = 0
                for stage in route:
                    _cost += stage.cost
                if cost > _cost:
                    cost = _cost
                    ret = p_id, p_edge, route

        if ret:
            self._parking_cache[edge] = ret
            return ret

        self.logger.fatal('Edge %s is not reachable from any parking lot.', edge)
        self._blacklisted_edges.add(edge)
        return None, None, None

    def _check_parkings_cache(self, edge):
        """ Check among the previously computed results of _find_closest_parking """
        if edge in self._parking_cache.keys():
            return self._parking_cache[edge]
        return None

####################################################################################################

#@ray.remote(num_cpus=10, num_gpus=1)
class CSParkingPersuasiveDeepMARLEnv(ComplexStochasticPersuasiveDeepMARLEnv):
    """ Simplified REWARD (with TRAVEL_TIME), aggregated MODE USAGE, FUTURE DEMAND and PARKING. """

    def __init__(self, config):
        """ Initialize the environment. """
        # SUMO Parking location
        self._load_parking_from_additionals(config['scenario_config']['misc']['sumo_parking_file'])
        super().__init__(config)

    def _load_parking_from_additionals(self, filename):
        """ Load parkings ids from XML file. """
        self._sumo_parking = collections.defaultdict(list) # edge => list of ID
        if not os.path.isfile(filename):
            raise Exception("config['scenario_config']['misc']['sumo_parking_file'] is missing")
        xml_tree = xml.etree.ElementTree.parse(filename).getroot()
        for child in xml_tree:
            if child.tag != 'parkingArea':
                continue
            edge = child.attrib['lane'].split('_')[0]
            self._sumo_parking[edge].append(child.attrib['id'])

    def _initialize_agents(self):
        self.agents = dict()
        self.waiting_agents = list()
        for agent, agent_config in self.agents_init_list.items():
            self.waiting_agents.append((agent_config.start, agent))
            self.agents[agent] = ComplexParkingDeepSUMOAgents(
                agent_config, self.sumo_net, self._sumo_parking)
        self.waiting_agents.sort()

####################################################################################################
