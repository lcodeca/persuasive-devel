#!/usr/bin/env python3

""" MARL Environment with parking enabled based on PersuasiveMultiAgentEnv. """

import collections
import logging
import os
import sys
import xml

from pprint import pformat

from environments.marlenvironment import PersuasiveMultiAgentEnv, SUMOModeAgent

# """ Import SUMO library """
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
    from traci.exceptions import TraCIException
    import traci.constants as tc
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

####################################################################################################

DEBUGGER = False
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

####################################################################################################

def env_creator(config):
    """ Environment creator used in the environment registration. """
    LOGGER.debug('[env_creator] Environment creation: MultiAgentEnvWParking')
    return MultiAgentEnvWParking(config)

####################################################################################################

#@ray.remote(num_cpus=10, num_gpus=1)
class SUMOAgentWParking(SUMOModeAgent):
    """ Agent implementation: mode decision. """

    # Class valiables containing the parking area info
    _edge_to_parking = collections.defaultdict(list)
    _parking_position = dict()
    _parking_cache = dict()

    def __init__(self, config, sumo_parking):
        super().__init__(config)
        self._load_parkings(sumo_parking)


    def _load_parkings(self, filename):
        """ Load parking ids from XML file. """
        xml_tree = xml.etree.ElementTree.parse(filename).getroot()
        for child in xml_tree:
            if child.tag != 'parkingArea':
                continue
            if child.attrib['id'] not in self._conf['intermodalOptions']['parkingAreaBlacklist']:
                edge = child.attrib['lane'].split('_')[0]
                position = float(child.attrib['startPos']) + 2.5
                self._edge_to_parking[edge].append(child.attrib['id'])
                self._parking_position[child.attrib['id']] = position

    def step(self, action, handler):
        """
        Implements the logic of each specific action passed as input.
        It's going to be called as following:
        for i, action in action_dict.items():
            obs[i], rew[i], done[i], info[i] = self.agents[i].step(action)
        """
        mode = None

        if action == 0:
            ### WAIT, nothing to do here.
            self.waited_steps += 1
            return False
        elif action in self.action_to_mode:
            mode = self.action_to_mode[action]
        else:
            raise NotImplementedError('Action {} is not implemented.'.format(action))

        self.chosen_mode = mode

        # compute the route using findIntermodalRoute
        _mode, _ptype, _vtype = handler.get_mode_parameters(mode)
        LOGGER.debug('Selected mode: %s. [mode %s, ptype %s, vtype %s]',
                     mode, _mode, _ptype, _vtype)
        try:
            route = handler.traci_handler.simulation.findIntermodalRoute(
                self.origin, self.destination, modes=_mode, pType=_ptype, vType=_vtype,
                routingMode=1)
            if not handler.is_valid_route(mode, route):
                route = None
        except TraCIException:
            route = None

        if route:
            try:
                # generate the person trip
                LOGGER.debug('Adding person %s', str(self.agent_id))
                # add(self, personID, edgeID, pos, depart=-3, typeID='DEFAULT_PEDTYPE')
                handler.traci_handler.person.add(self.agent_id, self.origin, 0.0)
                veh_counter = 0
                for stage in route:
                    self.cost += stage.cost
                    self.ett += stage.travelTime
                    # appendStage(self, personID, stage)
                    if DEBUGGER:
                        LOGGER.debug('%s', pformat(stage))
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
                        stage.line = vehicle_name
                        veh_counter += 1
                    handler.traci_handler.person.appendStage(self.agent_id, stage)
                return True
            except TraCIException:
                self.chosen_mode = None
                self.chosen_mode_error = 'TraCIException for mode {}'.format(mode)
                self.cost = float('NaN')
                self.ett = float('NaN')
                LOGGER.error('Route not usable for %s using mode %s', self.agent_id, mode)
                return True # wrong decision, paid badly at the end

        self.chosen_mode = None
        self.chosen_mode_error = 'Invalid route using mode {}'.format(mode)
        self.cost = float('NaN')
        self.ett = float('NaN')
        LOGGER.error('Route not found for %s using mode %s', self.agent_id, mode)
        return True # wrong decision, paid badly at the end

####################################################################################################

class MultiAgentEnvWParking(PersuasiveMultiAgentEnv):
    """
    MARL Environment with parking enabled based on PersuasiveMultiAgentEnv.
    https://github.com/ray-project/ray/blob/master/rllib/tests/test_multi_agent_env.py
    """

    def _initialize_agents(self):
        self.agents = dict()
        self.waiting_agents = list()
        for agent, agent_config in self.agents_init_list.items():
            self.waiting_agents.append((agent_config.start, agent))
            self.agents[agent] = SUMOAgentWParking(agent_config)
        self.waiting_agents.sort()
