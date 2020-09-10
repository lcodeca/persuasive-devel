#!/usr/bin/env python3

""" Persuasive MARL Environment based on RLLIB and SUMO """

import collections
import logging
import os
import sys

from pprint import pformat

import numpy as np
from numpy.random import RandomState

import gym

from ray.rllib.env import MultiAgentEnv
from rllibsumoutils.sumoutils import SUMOUtils, sumo_default_config

# """ Import SUMO library """
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
    import traci
    import libsumo
    import traci.constants as tc
    import sumolib
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

####################################################################################################

DEBUGGER = False
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('{}.log'.format(__name__)))
logger.setLevel(logging.INFO)

####################################################################################################

def env_creator(config):
    """ Environment creator used in the environment registration. """
    logger.debug('Environment creator --> PersuasiveMultiAgentEnv')
    return PersuasiveMultiAgentEnv(config)

#@ray.remote(num_cpus=10, num_gpus=1)
class SUMOSimulationWrapper(SUMOUtils):
    """ A wrapper for the interaction with the SUMO simulation """

    def _initialize_simulation(self):
        """ Specific simulation initialization. """
        try:
            super()._initialize_simulation()
        except NotImplementedError:
            pass

    def _initialize_metrics(self):
        """ Specific metrics initialization """
        try:
            super()._initialize_metrics()
        except NotImplementedError:
            pass

    def _default_step_action(self, agents):
        """ Specific code to be executed in every simulation step """
        try:
            super()._default_step_action(agents)
        except NotImplementedError:
            pass
        if agents:
            now = self.traci_handler.simulation.getTime() # seconds
            logger.debug('[%.2f] Agents: %s', now, str(agents))
            people = self.traci_handler.person.getIDList()
            logger.debug('[%.2f] People: %s', now, str(people))
            left = set()
            for agent in agents:
                if agent not in people:
                    left.add(agent)
            if len(agents) == len(left):
                # all the agents left the simulation
                logger.info('All the AGENTS left the SUMO simulation.')
                self.end_simulation()
        return True

    def get_penalty_time(self):
        """ Return the penalty max time. """
        return self._config['misc']['max_time']

####################################################################################################

#@ray.remote(num_cpus=10, num_gpus=1)
class SUMOModeAgent(object):
    """ Agent implementation: mode decision. """

    ## Configuration for the init
    ## {
    #       'origin': '-6040',
    #       'destination': '2370',
    #       'start': 21647,
    #       'expected-arrival-time': [32400, 2.0, 4.0],
    #       'modes': {'passenger': 1.0, 'public': 1.0},
    #       'seed': 321787075,
    #       'ext-stats': true,
    #  }
    Config = collections.namedtuple('Config',
                                    ['ID',              # Agent name
                                     'seed',            # Agent init random seed
                                     'ext',
                                     'start',           # Agent starting time
                                     'origin',          # edge - origin
                                     'destination',     # edge - destination
                                     'modes',           # list of available modes
                                     'exp_arrival'])    # expected arrival time and weight

    ## Actions --> Modes
    action_to_mode = {
        1: 'passenger',
        2: 'public',
        3: 'walk',
        4: 'bicycle',
        5: 'ptw',
        6: 'on-demand',
    }
    modes_w_vehicles = ['passenger', 'bicycle', 'ptw', 'on-demand',]

    def __init__(self, config):
        self._config = config
        self.agent_id = config.ID
        self.seed = config.seed
        self.ext = config.ext
        self.start = config.start
        self.origin = config.origin
        self.destination = config.destination
        self.modes = config.modes
        self.waited_steps = 0
        self.arrival, self.waiting_weight, self.late_weight = config.exp_arrival
        self.chosen_mode = None
        self.chosen_mode_error = None
        self.cost = 0.0
        self.ett = 0.0

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
        logger.debug('Selected mode: %s. [mode %s, ptype %s, vtype %s]',
                     mode, _mode, _ptype, _vtype)
        try:
            route = handler.traci_handler.simulation.findIntermodalRoute(
                self.origin, self.destination, modes=_mode, pType=_ptype, vType=_vtype,
                routingMode=1)
            if not handler.is_valid_route(mode, route):
                route = None
        except (traci.exceptions.TraCIException, libsumo.libsumo.TraCIException):
            route = None

        if route:
            try:
                # generate the person trip
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
                        stage.line = vehicle_name
                        veh_counter += 1
                    handler.traci_handler.person.appendStage(self.agent_id, stage)
                return True
            except (traci.exceptions.TraCIException, libsumo.libsumo.TraCIException) as exception:
                logger.error('%s', str(exception))
                self.chosen_mode = None
                self.chosen_mode_error = 'TraCIException for mode {}'.format(mode)
                self.cost = float('NaN')
                self.ett = float('NaN')
                logger.error('Route not usable for %s using mode %s', self.agent_id, mode)
                return True # wrong decision, paid badly at the end

        self.chosen_mode = None
        self.chosen_mode_error = 'Invalid route using mode {}'.format(mode)
        self.cost = float('NaN')
        self.ett = float('NaN')
        logger.error('Route not found for %s using mode %s', self.agent_id, mode)
        return True # wrong decision, paid badly at the end

    def reset(self):
        """ Resets the agent and return the observation. """
        self.waited_steps = 0
        self.chosen_mode = None
        self.chosen_mode_error = None
        self.cost = 0.0
        self.ett = 0.0
        return self.agent_id, self.start

    def __str__(self):
        """ Returns a string with all the internal state. """
        return('Config: {} \n Waited steps: {} \n Mode {} --> {} \n Cost: {} \n ETT: {} '.format(
            self._config, self.waited_steps, self.chosen_mode,
            self.chosen_mode_error, self.cost, self.ett))

####################################################################################################

DEFAULT_SCENARIO_CONFING = {
    'sumo_config': sumo_default_config(),
    'agent_rnd_order': True,
    'log_level': 'WARN',
    'seed': 42,
    'misc': {
        'sumo_net_file': '',
        'max_time': 86400
    }
}

DEFAULT_AGENT_CONFING = {
    'origin': '-6040',
    'destination': '2370',
    'start': 0,
    'expected-arrival-time': [32400, 2.0, 4.0],
        'modes': {
            'passenger': 1.0,
            'public': 1.0,
            'walk': 1.0,
            'bicycle': 1.0,
            'ptw': 1.0,
            'on-demand': 1.0
        },
    'seed': 42,
    'init': [0, 0],
    'ext-stats': False
}

class PersuasiveMultiAgentEnv(MultiAgentEnv):
    """
    Persuasive MARL environment for RLLIB SUMO Utils.

    https://github.com/ray-project/ray/blob/master/rllib/tests/test_multi_agent_env.py
    """

    def __init__(self, config):
        """ Initialize the environment. """
        super().__init__()

        logger.info('Environment creation: PersuasiveMultiAgentEnv')

        self._config = config
        self.metrics_dir = config['metrics_dir']
        self.checkpoint_dir = config['checkpoint_dir']

        # logging
        level = logging.getLevelName(config['scenario_config']['log_level'])
        logger.setLevel(level)

        # Random number generator
        self.rndgen = RandomState(config['scenario_config']['seed'])

        # SUMO Connector
        self.simulation = None

        # SUMO Network =
        self.sumo_net = sumolib.net.readNet(config['scenario_config']['misc']['sumo_net_file'])

        # Agent initialization
        self.agents_init_list = dict()
        self._config_from_agent_init()
        self._initialize_agents()
        self.dones = set()
        self.active = set()
        self.ext_stats = dict()

        # Environment initialization
        self.resetted = True
        self.environment_steps = 0
        self.episodes = 0
        self._edges_to_int = dict()
        self._ints_to_edge = dict()
        edges = self._load_relevant_edges()
        for pos, edge in enumerate(edges):
            self._edges_to_int[edge] = pos
            self._ints_to_edge[pos] = edge

    def _config_from_agent_init(self):
        """ Load the agents configuration and format it in a usabe format. """
        for agent, conf in self._config['agent_init'].items():
            if 'ext-stats' not in conf:
                conf['ext-stats'] = False
            self.agents_init_list[agent] = SUMOModeAgent.Config(
                agent, conf['seed'], conf['ext-stats'], conf['start'],
                conf['origin'], conf['destination'], conf['modes'],
                conf['expected-arrival-time'])
            if DEBUGGER:
                logger.debug('%s', pformat(self.agents_init_list[agent]))

    def _initialize_agents(self):
        self.agents = dict()
        self.waiting_agents = list()
        for agent, agent_config in self.agents_init_list.items():
            self.waiting_agents.append((agent_config.start, agent))
            self.agents[agent] = SUMOModeAgent(agent_config)
        self.waiting_agents.sort()

    def _load_relevant_edges(self):
        """ Returns a list containing all the edges in the SUMO NET file. """
        edges = list()
        for edge in self.sumo_net.getEdges():
            if ':' in edge.getID(): # no internal edges
                continue
            edges.append(edge.getID())
        logger.debug('Loaded %d edges', len(edges))
        return edges

    def seed(self, seed):
        """ Set the seed of a possible random number generator. """
        self.rndgen = RandomState(seed)

    def discrete_time(self, time_s):
        if np.isnan(time_s):
            return time_s
        # seconds to minutes, then slots of 2m
        minutes = self._config['scenario_config']['sumo_config']['update_freq'] / 60.0
        return int(round((time_s / 60.0 / minutes), 0))

    def get_agents(self):
        """ Returns a list of the agents. """
        return self.agents.keys()

    def __del__(self):
        logger.info('Environment destruction: PersuasiveMultiAgentEnv')
        if self.simulation:
            del self.simulation

    ################################################################################################

    def edge_to_enum(self, edge):
        """ Returns the integer associated with the given edge. """
        return self._edges_to_int[edge]

    def enum_to_edge(self, enum):
        """ Returns the edge associated with the given integer. """
        return self._ints_to_edge[enum]

    def craft_final_state(self, agent):
        final_state = collections.OrderedDict()
        final_state['from'] = self.edge_to_enum(self.agents[agent].origin)
        final_state['to'] = self.edge_to_enum(self.agents[agent].destination)
        final_state['time-left'] = self.discrete_time(0)
        final_state['ett'] = np.array([-1 for _ in self.agents[agent].modes])
        return final_state

    def _add_observed_ett(self, agent, mode, ett):
        """ Add the observation to the structure """
        if agent not in self.ext_stats:
            self.ext_stats[agent] = {mode: [ett],}
        elif mode not in self.ext_stats[agent]:
            self.ext_stats[agent][mode] = [ett]
        else:
            self.ext_stats[agent][mode].append(ett)

    def get_observation(self, agent):
        """
        Returns the observation of a given agent.
        Uses traci.simulation.findIntermodalRoute(
            fromEdge, toEdge, modes='', depart=-1.0, routingMode=0, speed=-1.0,
            walkFactor=-1.0, departPos=0.0, arrivalPos=-1073741824.0, departPosLat=0.0,
            pType='', vType='', destStop='')
        to retrieve the cost of a trip.
        [see http://sumo.sourceforge.net/pydoc/traci._simulation.html]
        """
        ett = []
        now = self.simulation.get_current_time()
        if DEBUGGER:
            logger.debug('Time: \n%s', pformat(now))
        origin = self.agents[agent].origin
        destination = self.agents[agent].destination
        for mode in sorted(self.agents[agent].modes):
            _mode, _ptype, _vtype = self.simulation.get_mode_parameters(mode)
            try:
                route = self.simulation.traci_handler.simulation.findIntermodalRoute(
                    origin, destination, modes=_mode, pType=_ptype, vType=_vtype, depart=now,
                    routingMode=1)
                if DEBUGGER:
                    logger.debug('SUMO Route: \n%s', pformat(route))
                if not self.simulation.is_valid_route(mode, route):
                    route = None
            except (traci.exceptions.TraCIException, libsumo.libsumo.TraCIException):
                route = None
            if DEBUGGER:
                logger.debug('Filtered route: \n%s', pformat(route))
            if route:
                cost = self.simulation.cost_from_route(route)
                ett.append(self.discrete_time(cost))
                if self.agents[agent].ext:
                    self._add_observed_ett(agent, mode, cost)
            else:
                ett.append(self.discrete_time(self._config['scenario_config']['misc']['max_time']))
                if self.agents[agent].ext:
                    self._add_observed_ett(
                        agent, mode, self._config['scenario_config']['misc']['max_time'])

        timeleft = self.agents[agent].arrival - now

        ret = collections.OrderedDict()
        ret['from'] = self.edge_to_enum(self.agents[agent].origin)
        ret['to'] = self.edge_to_enum(self.agents[agent].destination)
        ret['time-left'] = self.discrete_time(timeleft)
        ret['ett'] = np.array(ett)
        if DEBUGGER:
            logger.debug('Observation: \n%s', pformat(ret))
        return ret

    def compute_observations(self, agents):
        """ For each agent in the list, return the observation. """
        obs = dict()
        for agent in agents:
            obs[agent] = self.get_observation(agent)
        if DEBUGGER:
            logger.debug('Observations: \n%s', pformat(obs))
        return obs

    ################################################################################################

    def get_reward(self, agent):
        """ Return the reward for a given agent. """
        if not self.agents[agent].chosen_mode:
            logger.warning('Agent %s mode error: "%s"', agent, self.agents[agent].chosen_mode_error)
            return 0 - int(self.simulation.get_penalty_time())

        journey_time = self.simulation.get_duration(agent)
        if np.isnan(journey_time):
            ## This should never happen.
            ## If it does, there is a bug/issue with the SUMO/MARL environment interaction.
            raise Exception('{} \n {}'.format(
                self.compute_info_for_agent(agent), str(self.agents[agent])))
        logger.debug(' Agent: %s, journey: %s', agent, str(journey_time))
        arrival = self.simulation.get_arrival(agent,
                                              default=self.simulation.get_penalty_time())
        logger.debug(' Agent: %s, arrival: %s', agent, str(arrival))

        # REWARD = journey time * mode weight + ....
        reward = journey_time * self.agents[agent].modes[self.agents[agent].chosen_mode]
        if self.agents[agent].arrival < arrival:
            ## agent arrived too late
            late_time = arrival - self.agents[agent].arrival
            reward = self.simulation.get_penalty_time()
            logger.debug('Agent: %s, arrival: %s, wanted arrival: %s, late: %s',
                         agent, str(arrival), str(self.agents[agent].arrival), str(late_time))
        elif self.agents[agent].arrival > arrival:
            ## agent arrived too early
            waiting_time = self.agents[agent].arrival - arrival
            reward += waiting_time * self.agents[agent].waiting_weight
            logger.debug('Agent: %s, duration: %s, waiting: %s, wanted arrival: %s',
                         agent, str(journey_time), str(waiting_time), str(arrival))
        else:
            logger.debug('Agent: %s it is perfectly on time!', agent)

        return int(0 - (reward))

    def compute_rewards(self, agents):
        """ For each agent in the list, return the rewards. """

        # To compute the rewards from tripinfo data, the end of the simulation must be reached
        #    and the TraCI connection will be closed.
        self.simulation.process_tripinfo_file()

        rew = dict()
        for agent in agents:
            rew[agent] = self.get_reward(agent)
        return rew

    ################################################################################################

    def _move_agents(self, time):
        """ Move the agents from the waiting queue to the active queue. """
        if not self.waiting_agents:
            return False
        while self.waiting_agents[0][0] <= time:
            _, agent = self.waiting_agents.pop(0)
            self.active.add(agent)
            if not self.waiting_agents:
                return False
        return True

    def sumo_reset(self):
        return SUMOSimulationWrapper(self._config['scenario_config']['sumo_config'])

    def reset(self):
        """ Resets the env and returns observations from ready agents. """
        self.resetted = True
        self.environment_steps = 0
        self.episodes += 1
        self.active = set()
        self.dones = set()
        self.ext_stats = dict()

        # Reset the SUMO simulation
        if self.simulation:
            del self.simulation
        self.simulation = self.sumo_reset()

        # Reset the agents
        self.waiting_agents = list()
        for agent in self.agents.values():
            agent_id, start = agent.reset()
            self.waiting_agents.append((start, agent_id))
        self.waiting_agents.sort()

        # Move the simulation forward
        starting_time = self.waiting_agents[0][0]
        self.simulation.fast_forward(starting_time)
        self._move_agents(starting_time)

        # Observations
        initial_obs = self.compute_observations(self.active)

        return initial_obs

    def compute_info_for_agent(self, agent):
        """ Gather and return a dictionary containing the info associated with the given agent. """
        info = {
            'arrival': self.simulation.get_arrival(agent),
            'cost': self.agents[agent].cost,
            'departure': self.simulation.get_depart(agent),
            'discretized-cost': self.discrete_time(self.agents[agent].cost),
            'ett': self.agents[agent].ett,
            'mode': self.agents[agent].chosen_mode,
            'rtt': self.simulation.get_duration(agent),
            'timeLoss': self.simulation.get_timeloss(agent),
            'wait': self.agents[agent].arrival - self.simulation.get_arrival(agent)
        }
        logger.debug('Info for agent %s: \n%s', agent, pformat(info))
        return info

    def finalize_episode(self, dones):
        """ Gather all the required ifo and aggregate the rewards for the agents. """
        dones['__all__'] = True
        obs, rewards, infos = {}, {}, {}
        ## COMPUTE THE REWARDS
        rewards = self.compute_rewards(self.agents.keys())
        # FINAL STATE observations
        for agent in rewards:
            obs[agent] = self.craft_final_state(agent)
            infos[agent] = self.compute_info_for_agent(agent)
            if agent in self.ext_stats:
                infos[agent]['ext'] = self.ext_stats[agent]
        logger.debug('Observations: %s', str(obs))
        logger.debug('Rewards: %s', str(rewards))
        logger.debug('Dones: %s', str(dones))
        logger.debug('Info: %s', str(infos))
        return obs, rewards, dones, infos

    def step(self, action_dict):
        """
        Returns observations from ready agents.

        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.

        Returns
        -------
            obs (dict): New observations for each ready agent.
            rewards (dict): Reward values for each ready agent. If the
                episode is just started, the value will be None.
            dones (dict): Done values for each ready agent. The special key
                "__all__" (required) is used to indicate env termination.
            infos (dict): Optional info values for each agent id.
        """
        self.resetted = False
        self.environment_steps += 1
        logger.debug('====> [PersuasiveMultiAgentEnv:step] Episode: %d - Step: %d <====',
                     self.episodes, self.environment_steps)
        dones = {}

        shuffled_agents = sorted(action_dict.keys()) # it may seem not smar to sort something that
                                                     # may need to be shuffled afterwards, but it
                                                     # is a matter of consistency instead of using
                                                     # whatever insertion order was used in the dict
        if self._config['scenario_config']['agent_rnd_order']:
            ## randomize the agent order to minimize SUMO's insertion queues impact
            logger.debug('Shuffling the order of the agents.')
            self.rndgen.shuffle(shuffled_agents) # in-place shuffle

        # Take action
        for agent in shuffled_agents:
            dones[agent] = self.agents[agent].step(action_dict[agent], self.simulation)
            if dones[agent]:
                self.dones.add(agent)
                self.active.remove(agent)
        dones['__all__'] = len(self.dones) == len(self.agents)

        logger.debug('Before SUMO')
        until_end, agents_ids = False, set()
        if dones['__all__']:
            until_end = True
            agents_ids = set(self.agents.keys())
        ongoing_simulation = self.simulation.step(until_end=until_end, agents=agents_ids)
        logger.debug('After SUMO')

        current_time = self.simulation.get_current_time()

        ## end of the episode
        if not ongoing_simulation:
            logger.debug('Reached the end of the SUMO simulation. Finalizing episode...')
            return self.finalize_episode(dones)

        ## add waiting agent to the pool of active agents
        self._move_agents(self.simulation.get_current_time())

        # compute the new observation for the WAITING agents
        logger.debug('Computing obseravions for the WAITING agents.')
        obs, rewards, infos = {}, {}, {}
        agents_to_remove = set()
        for agent in self.active:
            logger.debug('[%2f] %s --> %d', current_time, agent, self.agents[agent].arrival)
            if current_time > self.agents[agent].arrival:
                logger.warning('Agent %s waited for too long.', str(agent))
                self.agents[agent].chosen_mode_error = 'Waiting too long [{}]'.format(current_time)
                self.agents[agent].ett = float('NaN')
                self.agents[agent].cost = float('NaN')
                dones[agent] = True
                self.dones.add(agent)
                agents_to_remove.add(agent)
            else:
                rewards[agent] = 0
                obs[agent] = self.get_observation(agent)
                infos[agent] = {}
        for agent in agents_to_remove:
            self.active.remove(agent)

        # in case all the reamining agents WAITED TOO LONG
        dones['__all__'] = len(self.dones) == len(self.agents)
        if dones['__all__']:
            logger.info('All the agent are DONE. Finalizing episode...')
            self.simulation.step(until_end=True, agents=set(self.agents.keys()))
            return self.finalize_episode(dones)

        logger.debug('Observations: %s', str(obs))
        logger.debug('Rewards: %s', str(rewards))
        logger.debug('Dones: %s', str(dones))
        logger.debug('Info: %s', str(infos))
        logger.debug('========================================================')
        return obs, rewards, dones, infos

    def get_environment_steps(self):
        """ Returns the total number of SUMOSimulation.step() """
        return self.environment_steps

    ################################################################################################

    def get_action_space_size(self, agent):
        """ Returns the size of the action space. """
        return len(self.agents[agent].modes) + 1 # + WAIT state

    def get_action_space(self, agent):
        """ Returns the action space. """
        return gym.spaces.Discrete(self.get_action_space_size(agent))

    def get_set_of_actions(self, agent):
        """ Returns the set of possible actions for an agent. """
        return set(range(self.get_action_space_size(agent)))

    def get_obs_space_size(self, agent):
        """ Returns the size of the observation space. """
        _from = len(self._edges_to_int)
        _to = len(self._edges_to_int)
        _time_left = self._config['scenario_config']['misc']['max_time']
        _ett = (
            self._config['scenario_config']['misc']['max_time'] * len(self.agents[agent].modes))
        return _from * _to * _time_left * _ett

    def get_obs_space(self, agent):
        """ Returns the observation space. """
        return gym.spaces.Dict({
            'from': gym.spaces.Discrete(len(self._edges_to_int)),
            'to': gym.spaces.Discrete(len(self._edges_to_int)),
            'time-left': gym.spaces.Discrete(self._config['scenario_config']['misc']['max_time']),
            'ett': gym.spaces.MultiDiscrete(
                [self._config['scenario_config']['misc']['max_time']] * (
                    len(self.agents[agent].modes))),
        })

    def get_metrics(self):
        """ Returns the custom experiment metrics. """
        metrics = collections.defaultdict(list)
        for vehicle in self.simulation.tripinfo.values():
            for metric, value in vehicle.items():
                try:
                    val = float(value)
                    metrics[metric].append(val)
                except ValueError:
                    pass
        for person in self.simulation.personinfo.values():
            for metric, value in person.items():
                if 'stages' in metric:
                    for _, stage in value:
                        for _metric, _value in stage.items():
                            try:
                                val = float(_value)
                                metrics[_metric].append(val)
                            except ValueError:
                                pass
                else:
                    try:
                        val = float(value)
                        metrics[metric].append(val)
                    except ValueError:
                        pass
        summary = dict()
        for metric, values in metrics.items():
            summary[metric] = {
                'min': np.min(values),
                'max': np.max(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
            }
        return self.metrics_dir, summary

    ################################################################################################
    #        OLD VERSIONS OF REWARDS
    ################################################################################################

    def get_reward_duration_wait_late_linear(self, agent):
        """ Return the reward for a given agent. """
        if not self.agents[agent].chosen_mode:
            logger.error('Agent %s selected a mode that was not possible.', agent)
            return 0 - int(self.simulation.get_penalty_time() * self.agents[agent].late_weight)

        journey_time = self.simulation.get_duration(agent)
        logger.debug(' Agent: %s, journey: %s', agent, str(journey_time))
        arrival = self.simulation.get_arrival(agent)
        logger.debug(' Agent: %s, arrival: %s', agent, str(arrival))

        # REWARD = journey time * mode weight + ....
        reward = journey_time * self.agents[agent].modes[self.agents[agent].chosen_mode]
        if self.agents[agent].arrival < arrival:
            ## agent arrived too late
            late_time = arrival - self.agents[agent].arrival
            reward += late_time * self.agents[agent].late_weight
            logger.debug('Agent: %s, arrival: %s, wanted arrival: %s, late: %s',
                         agent, str(arrival), str(self.agents[agent].arrival), str(late_time))
        elif self.agents[agent].arrival > arrival:
            ## agent arrived too early
            waiting_time = self.agents[agent].arrival - arrival
            reward += waiting_time * self.agents[agent].waiting_weight
            logger.debug('Agent: %s, duration: %s, waiting: %s, wanted arrival: %s',
                         agent, str(journey_time), str(waiting_time), str(arrival))
        else:
            logger.debug('Agent: %s it is perfectly on time!', agent)

        return int(0 - (reward))

    def get_reward_duration_wait_late_exp(self, agent):
        """ Return the reward for a given agent. """
        if not self.agents[agent].chosen_mode:
            logger.error('Agent %s selected a mode that was not possible.', agent)
            return 0 - int(self.simulation.get_penalty_time() ** self.agents[agent].late_weight)

        journey_time = self.simulation.get_duration(agent)
        logger.debug(' Agent: %s, journey: %s', agent, str(journey_time))
        arrival = self.simulation.get_arrival(agent)
        logger.debug(' Agent: %s, arrival: %s', agent, str(arrival))

        # REWARD = journey time * mode weight + ....
        reward = journey_time * self.agents[agent].modes[self.agents[agent].chosen_mode]
        if self.agents[agent].arrival < arrival:
            ## agent arrived too late
            late_time = arrival - self.agents[agent].arrival
            reward += late_time ** self.agents[agent].late_weight
            logger.error('Agent: %s, arrival: %s, wanted arrival: %s, late: %s',
                         agent, str(arrival), str(self.agents[agent].arrival), str(late_time))
        elif self.agents[agent].arrival > arrival:
            ## agent arrived too early
            waiting_time = self.agents[agent].arrival - arrival
            reward += waiting_time ** self.agents[agent].waiting_weight
            logger.error('Agent: %s, duration: %s, waiting: %s, wanted arrival: %s',
                         agent, str(journey_time), str(waiting_time), str(arrival))
        else:
            logger.error('Agent: %s it is perfectly on time!', agent)

        return int(0 - (reward))

    ################################################################################################
    #        OLD VERSIONS OF OBSERVATIONS
    ################################################################################################

    # From EDGE --> To EDGE --> only ranking by mode.
    def get_rank_obs_space_size(self, agent):
        """ Returns the size of the observation space. """
        return (len(self._edges_to_int) *                                          # from
                len(self._edges_to_int) *                                          # to
                len(self.agents[agent].modes) * len(self.agents[agent].modes))     # rank

    def get_rank_obs_space(self, agent):
        """ Returns the observation space. """
        return gym.spaces.Dict({
            'from': gym.spaces.Discrete(len(self._edges_to_int)),
            'to': gym.spaces.Discrete(len(self._edges_to_int)),
            'rank': gym.spaces.MultiDiscrete(
                [len(self.agents[agent].modes)] * (len(self.agents[agent].modes))),
        })

    def craft_rank_final_state(self, agent):
        final_state = collections.OrderedDict()
        final_state['from'] = self.edge_to_enum(self.agents[agent].origin)
        final_state['to'] = self.edge_to_enum(self.agents[agent].destination)
        final_state['rank'] = np.array([-1 for _ in self.agents[agent].modes])
        return final_state

    def get_rank_observation(self, agent):
        """
        Returns the observation of a given agent.
        Uses traci.simulation.findIntermodalRoute(
            fromEdge, toEdge, modes='', depart=-1.0, routingMode=0, speed=-1.0,
            walkFactor=-1.0, departPos=0.0, arrivalPos=-1073741824.0, departPosLat=0.0,
            pType='', vType='', destStop='')
        to retrieve the cost of a trip.
        [see http://sumo.sourceforge.net/pydoc/traci._simulation.html]
        """
        ett = []
        origin = self.agents[agent].origin
        destination = self.agents[agent].destination
        for mode in sorted(self.agents[agent].modes):
            _mode, _ptype, _vtype = self.simulation.get_mode_parameters(mode)
            try:
                route = self.simulation.traci_handler.simulation.findIntermodalRoute(
                    origin, destination, modes=_mode, pType=_ptype, vType=_vtype, routingMode=1)
                if not self.simulation.is_valid_route(mode, route):
                    route = None
            except (traci.exceptions.TraCIException, libsumo.libsumo.TraCIException):
                route = None
            if route:
                ett.append((float(self.simulation.cost_from_route(route)), mode))
            else:
                ett.append((self._config['scenario_config']['misc']['max_time'], mode))

        obs = []
        ett = [mode for _, mode in sorted(ett)]
        for mode in sorted(self.agents[agent].modes):
            obs.append(ett.index(mode))

        ret = collections.OrderedDict()
        ret['from'] = self.edge_to_enum(self.agents[agent].origin)
        ret['to'] = self.edge_to_enum(self.agents[agent].destination)
        ret['rank'] = np.array(obs)
        if DEBUGGER:
            logger.debug('Observation: \n%s', pformat(ret))
        return ret

    ################################################################################################
