import os
import sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import sumolib
from gym import Env
import traci.constants as tc
from gym import spaces
import numpy as np
import pandas as pd

# from PIL import Image

import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from platoon_src.simulationmanager import SimulationManager
from platoon_src.simlib import setUpSimulation

from .traffic_signal import TrafficSignal


class PlatoonEnvironment:
    """
    SUMO Environment for Traffic Signal Control

    :param net_file: (str) SUMO .net.xml file
    :param route_file: (str) SUMO .rou.xml file
    :param phases: (traci.trafficlight.Phase list) Traffic Signal phases definition
    :param out_csv_name: (str) name of the .csv output with simulation results. If None no output is generated
    :param use_gui: (bool) Wheter to run SUMO simulation with GUI visualisation
    :param num_seconds: (int) Number of simulated seconds on SUMO
    :param max_depart_delay: (int) Vehicles are discarded if they could not be inserted after max_depart_delay seconds
    :param time_to_load_vehicles: (int) Number of simulation seconds ran before learning begins
    :param delta_time: (int) Simulation seconds between actions
    :param min_green: (int) Minimum green time in a phase
    :param max_green: (int) Max green time in a phase
    """

    def __init__(self, net_file, route_file, cfg_file, phases, out_csv_name=None, out_csv_name2=None, use_gui=False,
                 direct_start=True, num_seconds=20000,
                 max_depart_delay=100000, time_to_load_vehicles=0, num_green_phases=8, delta_time=5, min_green=5,
                 max_green=100):

        ##### self defined phases, need to comment out self.pahses = phases
        # def __init__(self, net_file, route_file, cfg_file, out_csv_name=None, out_csv_name2=None, use_gui=False, direct_start=True, num_seconds=20000,
        #              max_depart_delay=100000, time_to_load_vehicles=0, num_green_phases=8, delta_time=5, min_green=5, max_green=100): #self defined phases

        self._net = net_file
        self._route = route_file
        self._cfg = cfg_file
        self.use_gui = use_gui
        self.direct_start = direct_start
        if self.use_gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')

        traci.start([sumolib.checkBinary('sumo'), '-n', self._net])  # start only to retrieve information

        self.ts_ids = traci.trafficlight.getIDList()
        self.lanes_per_ts = len(set(traci.trafficlight.getControlledLanes(self.ts_ids[0])))
        self.traffic_signals = dict()
        self.phases = phases
        self.num_green_phases = num_green_phases  # len(phases) // 2  # Number of green phases == number of phases (green+yellow) divided by 2
        self.vehicles = dict()
        self.last_measure = dict()  # used to reward function remember last measure
        self.last_reward = 0
        self.sim_max_time = num_seconds
        self.time_to_load_vehicles = time_to_load_vehicles  # number of simulation seconds ran in reset() before learning starts
        self.delta_time = delta_time  # seconds on sumo at each step
        self.max_depart_delay = max_depart_delay  # Max wait time to insert a vehicle
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = 2
        """
        self.state_count = 0
        """
        self.manager = SimulationManager(True, False)

        self.observation_space = (60,)

        self.action_space = 8  # Split, Speedup, Slow down, and No-op
        """
        self.action_space = (self.max_platoon_speed - self.min_platoon_speed + 1) * (self.max_platoon_member - self.min_platoon_member + 1)
        """
        self.reward_range = (-float('inf'), float('inf'))
        self.metadata = {}
        self.spec = ''

        # self.radix_factors = [s.n for s in self.discrete_observation_space.spaces]
        # self.run = 0
        self.metrics = []
        self.trajectory = []
        """
        record action info
        """
        self.action_info = []
        self.out_csv_name = out_csv_name
        self.out_csv_name2 = out_csv_name2

        traci.close()

    def reset(self):
        """
        if self.run != 0:
            self.save_csv(self.out_csv_name, self.run)
        self.run += 1
        """
        self.metrics = []
        self.trajectory = []
        """
        setUpSimulation(self._cfg, self._sumo_binary, 3)
        """
        sumo_cmd = [self._sumo_binary,
                    '-n', self._net,
                    '-r', self._route,
                    '--quit-on-end',
                    "--collision.action", "none",
                    '--max-depart-delay', str(self.max_depart_delay),
                    '--waiting-time-memory', '10000',
                    '--random',
                    '--log', '../outputs/RL_platoon/sumo_step_info.txt',
                    '--no-step-log',
                    '--no-warnings']
        if self.direct_start:
            sumo_cmd.append('--start')
        traci.start(sumo_cmd)

        for ts in self.ts_ids:
            #     if ts == "m1":
            #         phases=[
            #             traci.trafficlight.Phase(20, "GGGgrrrrGGGgrrrr"),
            #             traci.trafficlight.Phase(2, "yyyyrrrryyyyrrrr"),
            #             traci.trafficlight.Phase(20, "rrrrGGGgrrrrGGGg"),
            #             traci.trafficlight.Phase(2, "rrrryyyyrrrryyyy")
            #         ]
            #     elif ts == "m2":
            #         phases=[
            #             traci.trafficlight.Phase(20, "GGGgrrrrGGGgrrrr"),
            #             traci.trafficlight.Phase(2, "yyyyrrrryyyyrrrr"),
            #             traci.trafficlight.Phase(20, "rrrrGGGgrrrrGGGg"),
            #             traci.trafficlight.Phase(2, "rrrryyyyrrrryyyy")
            #         ]
            #     elif ts == "m3":
            #         phases=[
            #             traci.trafficlight.Phase(18, "GGGgrrrrGGGgrrrr"),
            #             traci.trafficlight.Phase(2, "yyyyrrrryyyyrrrr"),
            #             traci.trafficlight.Phase(20, "rrrrGGGgrrrrGGGg"),
            #             traci.trafficlight.Phase(2, "rrrryyyyrrrryyyy")
            #         ]
            #     elif ts == "n1":
            #         phases=[
            #             traci.trafficlight.Phase(18, "GGGgrrrrGGGgrrrr"),
            #             traci.trafficlight.Phase(2, "yyyyrrrryyyyrrrr"),
            #             traci.trafficlight.Phase(22, "rrrrGGGgrrrrGGGg"),
            #             traci.trafficlight.Phase(2, "rrrryyyyrrrryyyy")
            #         ]
            #     elif ts == "n2":
            #         phases=[
            #             traci.trafficlight.Phase(22, "GGGgrrrrGGGgrrrr"),
            #             traci.trafficlight.Phase(2, "yyyyrrrryyyyrrrr"),
            #             traci.trafficlight.Phase(22, "rrrrGGGgrrrrGGGg"),
            #             traci.trafficlight.Phase(2, "rrrryyyyrrrryyyy")
            #         ]
            #     elif ts == "n3":
            #         phases=[
            #             traci.trafficlight.Phase(21, "GGGgrrrrGGGgrrrr"),
            #             traci.trafficlight.Phase(2, "yyyyrrrryyyyrrrr"),
            #             traci.trafficlight.Phase(19, "rrrrGGGgrrrrGGGg"),
            #             traci.trafficlight.Phase(2, "rrrryyyyrrrryyyy")
            #         ]
            #     elif ts == "s1":
            #         phases=[
            #             traci.trafficlight.Phase(21, "GGGgrrrrGGGgrrrr"),
            #             traci.trafficlight.Phase(2, "yyyyrrrryyyyrrrr"),
            #             traci.trafficlight.Phase(19, "rrrrGGGgrrrrGGGg"),
            #             traci.trafficlight.Phase(2, "rrrryyyyrrrryyyy")
            #         ]
            #     elif ts == "s2":
            #         phases=[
            #             traci.trafficlight.Phase(22, "GGGgrrrrGGGgrrrr"),
            #             traci.trafficlight.Phase(2, "yyyyrrrryyyyrrrr"),
            #             traci.trafficlight.Phase(22, "rrrrGGGgrrrrGGGg"),
            #             traci.trafficlight.Phase(2, "rrrryyyyrrrryyyy")
            #         ]
            #     else:
            #         phases=[
            #             traci.trafficlight.Phase(18, "GGGgrrrrGGGgrrrr"),
            #             traci.trafficlight.Phase(2, "yyyyrrrryyyyrrrr"),
            #             traci.trafficlight.Phase(22, "rrrrGGGgrrrrGGGg"),
            #             traci.trafficlight.Phase(2, "rrrryyyyrrrryyyy")
            #         ]
            #
            # for ts in self.ts_ids:
            #     if ts == "m1":
            #         phases=[
            #             traci.trafficlight.Phase(22, "GGGgrrrrGGGgrrrr"),
            #             traci.trafficlight.Phase(22, "rrrrGGGgrrrrGGGg")
            #         ]
            #     elif ts == "m2":
            #         phases=[
            #             traci.trafficlight.Phase(22, "GGGgrrrrGGGgrrrr"),
            #             traci.trafficlight.Phase(22, "rrrrGGGgrrrrGGGg")
            #         ]
            #     elif ts == "m3":
            #         phases=[
            #             traci.trafficlight.Phase(20, "GGGgrrrrGGGgrrrr"),
            #             traci.trafficlight.Phase(22, "rrrrGGGgrrrrGGGg")
            #         ]
            #     elif ts == "n1":
            #         phases=[
            #             traci.trafficlight.Phase(20, "GGGgrrrrGGGgrrrr"),
            #             traci.trafficlight.Phase(24, "rrrrGGGgrrrrGGGg")
            #         ]
            #     elif ts == "n2":
            #         phases=[
            #             traci.trafficlight.Phase(24, "GGGgrrrrGGGgrrrr"),
            #             traci.trafficlight.Phase(64, "rrrrGGGgrrrrGGGg")
            #         ]
            #     elif ts == "n3":
            #         phases=[
            #             traci.trafficlight.Phase(23, "GGGgrrrrGGGgrrrr"),
            #             traci.trafficlight.Phase(21, "rrrrGGGgrrrrGGGg")
            #         ]
            #     elif ts == "s1":
            #         phases=[
            #             traci.trafficlight.Phase(23, "GGGgrrrrGGGgrrrr"),
            #             traci.trafficlight.Phase(21, "rrrrGGGgrrrrGGGg")
            #         ]
            #     elif ts == "s2":
            #         phases=[
            #             traci.trafficlight.Phase(24, "GGGgrrrrGGGgrrrr"),
            #             traci.trafficlight.Phase(64, "rrrrGGGgrrrrGGGg")
            #         ]
            #     else:
            #         phases=[
            #             traci.trafficlight.Phase(20, "GGGgrrrrGGGgrrrr"),
            #             traci.trafficlight.Phase(24, "rrrrGGGgrrrrGGGg")
            #         ]

            self.traffic_signals[ts] = TrafficSignal(self, ts, self.delta_time, self.min_green, self.max_green,
                                                     self.phases, self.num_green_phases)

            ##### For self defined phases
            # self.traffic_signals[ts] = TrafficSignal(self, ts, self.delta_time, self.min_green, self.max_green, phases, self.num_green_phases) # self-defined phases
            self.last_measure[ts] = 0.0

            # print(self.traffic_signals[ts].edges)

        self.vehicles = dict()

        # Load vehicles
        for _ in range(self.time_to_load_vehicles):
            self._sumo_step()
            self._compute_trajectory_info()
        return self._compute_observations()

    """ ccyen """

    def no_op_step(self, time_to_operate):
        for _ in range(time_to_operate):
            self._sumo_step()
            self._compute_trajectory_info()

    @property
    def sim_step(self):
        """
        Return current simulation second on SUMO
        """
        return traci.simulation.getCurrentTime() / 1000  # milliseconds to seconds

    def step(self, joint_actions):
        self._apply_actions(joint_actions)
        self._compute_action_info(joint_actions)

        for _ in range(self.yellow_time):
            self._sumo_step()
            self._compute_trajectory_info()
        """
        for ts in self.ts_ids:
            self.traffic_signals[ts].update_phase()
        """
        for _ in range(self.delta_time - self.yellow_time):
            self._sumo_step()
            self._compute_trajectory_info()

        # observe new state and reward
        observation = self._compute_observations()
        reward = self._compute_rewards()
        done = {'__all__': self.sim_step > self.sim_max_time}
        info = self._compute_step_info()
        self.metrics.append(info)
        self.last_reward = reward

        return observation, reward, done, info

    def _apply_actions(self, joint_actions):
        """
        Set the maximum number of vehicles in a platoon for the platoon manager
        """

        allVehicles = traci.vehicle.getIDList()

        for p in self.manager.getActivePlatoons():
            if p.getID() in allVehicles and p.getID() in joint_actions:

                # Split
                if joint_actions[p.getID()] == 0:
                    self.manager.setReform(p, True)
                # Speedup
                elif joint_actions[p.getID()] == 1:
                    new_speed = p.getSpeed() + 5
                    self.manager.setMaxSpeedForAPlat(p, new_speed)
                elif joint_actions[p.getID()] == 2:
                    new_speed = p.getSpeed() + 3
                    self.manager.setMaxSpeedForAPlat(p, new_speed)
                elif joint_actions[p.getID()] == 3:
                    new_speed = p.getSpeed() + 1
                    self.manager.setMaxSpeedForAPlat(p, new_speed)
                # Slow down
                elif joint_actions[p.getID()] == 3:
                    new_speed = max(p.getSpeed() - 5, 1)
                    self.manager.setMaxSpeedForAPlat(p, new_speed)
                elif joint_actions[p.getID()] == 5:
                    new_speed = max(p.getSpeed() - 3, 1)
                    self.manager.setMaxSpeedForAPlat(p, new_speed)
                elif joint_actions[p.getID()] == 6:
                    new_speed = max(p.getSpeed() - 1, 1)
                    self.manager.setMaxSpeedForAPlat(p, new_speed)
                # NO-OP
                else:
                    # Do nothing
                    xxx = 1

                #
                # Split
                # if joint_actions[p.getID()] == 0:
                #     xxx = 1
                # # NO-OP
                # else:
                #     # Do nothing
                #     yyy = 1

    def _compute_observations(self):
        """
        Return the current observation for each traffic signal
        """

        """ ccyen """
        allVehicles = traci.vehicle.getIDList()
        platoon_arrival_vec = {}

        for p in self.manager.getActivePlatoons():
            if p.getID() in allVehicles:
                member_list = p.getAllVehicles()
                platoon_arrival_vec[p.getID()] = self.manager.get_timing(member_list)
        # print(platoon_arrival_vec)
        return platoon_arrival_vec

    """ Hang """

    def _compute_trajectory_info(self):
        allVehicles = traci.vehicle.getIDList()
        for v in allVehicles:
            step_time = self.sim_step
            ID = v
            position = traci.vehicle.getPosition(v)
            speed = traci.vehicle.getSpeed(v)
            acc = traci.vehicle.getAcceleration(v)
            # if v in joint_actions:
            #     if joint_actions[v] == []:
            #         act = -1
            #     else:
            #         act = joint_actions[v]

            info_vehicle = {'step_time': step_time,
                            'vehicle_ID': ID,
                            'position': position,
                            'speed': speed,
                            'acceleration': acc
                            # 'action' : act
                            }
            self.trajectory.append(info_vehicle)

    " Hang "

    def _compute_action_info(self, joint_actions):
        allVehicles = traci.vehicle.getIDList()
        step_time = self.sim_step

        for p in self.manager.getActivePlatoons():
            if p.getID() in allVehicles and p.getID() in joint_actions:
                ID = p.getID()
                action = joint_actions[p.getID()]
                minGap = traci.vehicle.getMinGap(p.getID())
                info_platoon_action = {
                    'step_time': step_time,
                    'vehicle_ID': ID,
                    'action': action,
                    'Gap': minGap
                }
                self.action_info.append(info_platoon_action)

    def _compute_rewards(self):

        # return self._power_metric_reward()
        # return self._stopped_vehicle_reward() * -1
        # return self._fuel_consumption_reward() * -1
        # return self._waiting_time_reward()
        # return self._queue_reward()
        # return self._waiting_time_reward2()
        # return self._queue_average_reward()

        # return self._power_metric_by_platoon_reward()
        # return self._power_metric_by_platoon_reward_2()
        # return self._power_metric_by_platoon_reward_3()
        return self._moving_vehicles_by_platoon_reward()
        # return self._stopped_vehicles_by_platoon_reward()
        # return self._fuel_consumption_by_platoon_reward()
        # return self._no_reward()

    def _queue_average_reward(self):
        rewards = {}
        for ts in self.ts_ids:
            new_average = np.mean(self.traffic_signals[ts].get_stopped_vehicles_num())
            rewards[ts] = self.last_measure[ts] - new_average
            self.last_measure[ts] = new_average
        return rewards

    def _queue_reward(self):
        rewards = {}
        for ts in self.ts_ids:
            rewards[ts] = - (sum(self.traffic_signals[ts].get_stopped_vehicles_num())) ** 2
        return rewards

    def _waiting_time_reward(self):
        rewards = {}
        for ts in self.ts_ids:
            ts_wait = sum(self.traffic_signals[ts].get_waiting_time())
            rewards[ts] = self.last_measure[ts] - ts_wait
            self.last_measure[ts] = ts_wait
        return rewards

    def _waiting_time_reward2(self):
        rewards = {}
        for ts in self.ts_ids:
            ts_wait = sum(self.traffic_signals[ts].get_waiting_time())
            self.last_measure[ts] = ts_wait
            if ts_wait == 0:
                rewards[ts] = 1.0
            else:
                rewards[ts] = 1.0 / ts_wait
        return rewards

    def _waiting_time_reward3(self):
        rewards = {}
        for ts in self.ts_ids:
            ts_wait = sum(self.traffic_signals[ts].get_waiting_time())
            rewards[ts] = -ts_wait
            self.last_measure[ts] = ts_wait
        return rewards

    """ Hang """

    def _no_reward(self):
        rewards = {}
        allVehicles = traci.vehicle.getIDList()

        for p in self.manager.getActivePlatoons():
            if p.getID() in allVehicles:
                rewards[p.getID()] = 0

        return rewards

    """ ccyen """

    def _fuel_consumption_reward(self):
        rewards = 0
        for ts in self.ts_ids:
            ts_fuel = sum(self.traffic_signals[ts].get_fuel_consumption())
            rewards = rewards + ts_fuel
        return rewards

    """ ccyen """

    def _power_metric_reward(self):
        throughput = 0
        if len(self.metrics) > 1:
            for veh in self.metrics[-2]['total_vehicle_list']:
                if veh not in self.metrics[-1]['total_vehicle_list']:
                    throughput += 1
        fuel = self._fuel_consumption_reward()
        return throughput / fuel

    def _stopped_vehicle_reward(self):
        rewards = 0
        for ts in self.ts_ids:
            ts_stopped_veh = sum(self.traffic_signals[ts].get_stopped_vehicles_num())
            rewards = rewards + ts_stopped_veh
        return rewards

    """ ccyen """

    def _fuel_consumption_by_platoon_reward(self):
        rewards = {}
        allVehicles = traci.vehicle.getIDList()

        for p in self.manager.getActivePlatoons():
            if p.getID() in allVehicles:
                rewards[p.getID()] = 0
                for v in p._vehicles:
                    if v.getName() in allVehicles:
                        rewards[p.getID()] += traci.vehicle.getFuelConsumption(v.getName()) * -1
        return rewards

    """ ccyen """

    def _power_metric_by_platoon_reward(self):
        rewards = {}
        allVehicles = traci.vehicle.getIDList()

        for p in self.manager.getActivePlatoons():
            if p.getID() in allVehicles:
                throughput = 0
                avg_fuel = 1
                for v in p._vehicles:
                    if v.getName() in allVehicles:
                        nextInt = traci.vehicle.getNextTLS(v.getName())
                        if not nextInt:
                            throughput += 1

                        avg_fuel += traci.vehicle.getFuelConsumption(v.getName())

                avg_fuel = avg_fuel / (len(p._vehicles))

                rewards[p.getID()] = throughput / avg_fuel

        return rewards

    """ ccyen """

    def _power_metric_by_platoon_reward_2(self):
        rewards = {}
        allVehicles = traci.vehicle.getIDList()

        for p in self.manager.getActivePlatoons():
            if p.getID() in allVehicles:
                moving_veh = 0
                avg_delay = 1
                for v in p._vehicles:
                    if v.getName() in allVehicles:
                        if v.getSpeed() >= 1:
                            moving_veh += 1

                        avg_delay += traci.vehicle.getAccumulatedWaitingTime(v.getName())

                avg_delay = avg_delay / (len(p._vehicles))

                rewards[p.getID()] = moving_veh / avg_delay

        return rewards

    """ ccyen """

    def _power_metric_by_platoon_reward_3(self):
        rewards = {}
        allVehicles = traci.vehicle.getIDList()

        for p in self.manager.getActivePlatoons():
            if p.getID() in allVehicles:
                moving_veh = 0
                avg_fuel = 1
                for v in p._vehicles:
                    if v.getName() in allVehicles:
                        if v.getSpeed() >= 1:
                            moving_veh += 1

                        avg_fuel += traci.vehicle.getFuelConsumption(v.getName())

                avg_fuel = avg_fuel / (len(p._vehicles))

                rewards[p.getID()] = moving_veh / avg_fuel

        return rewards

    """ ccyen """

    def _stopped_vehicles_by_platoon_reward(self):
        rewards = {}
        allVehicles = traci.vehicle.getIDList()

        for p in self.manager.getActivePlatoons():
            if p.getID() in allVehicles:
                rewards[p.getID()] = 0
                for v in p._vehicles:
                    if v.getName() in allVehicles and v.getSpeed() < 1:
                        rewards[p.getID()] += -1
        return rewards

    """ ccyen """

    def _moving_vehicles_by_platoon_reward(self):
        rewards = {}
        allVehicles = traci.vehicle.getIDList()

        for p in self.manager.getActivePlatoons():
            if p.getID() in allVehicles:
                moving_veh = 0
                for v in p._vehicles:
                    if v.getName() in allVehicles:
                        if v.getSpeed() >= 15:
                            moving_veh += 1

                rewards[p.getID()] = moving_veh

        return rewards

    def _sumo_step(self):
        self.manager.handleSimulationStep()
        traci.simulationStep()

    def _compute_step_info(self):
        allVehicles = traci.vehicle.getIDList()
        fuel = 0
        delay = 0
        num_v = 1
        for v in allVehicles:
            fuel += traci.vehicle.getFuelConsumption(v)
            delay += traci.vehicle.getAccumulatedWaitingTime(v)
            num_v += 1

        return {
            'step_time': self.sim_step,
            'reward': self.last_reward,
            'total_stopped': sum([sum(self.traffic_signals[ts].get_stopped_vehicles_num()) for ts in self.ts_ids]),
            # 'total_wait_time': sum([self.last_measure[ts] for ts in self.ts_ids]),
            # 'total_wait_time': sum([sum(self.traffic_signals[ts].get_waiting_time()) for ts in self.ts_ids]),
            'avg_wait_time': delay / num_v,
            'total_vehicle_list': traci.vehicle.getIDList(),
            # 'total_fuel_consumption': sum([sum(self.traffic_signals[ts].get_fuel_consumption()) for ts in self.ts_ids])
            'avg_fuel_consumption': fuel / num_v
        }

    def close(self):
        traci.close()

    def encode(self, state):
        phase = state[:self.num_green_phases].index(1)
        elapsed = self._discretize_elapsed_time(state[self.num_green_phases])
        density_queue = [self._discretize_density(d) for d in state[self.num_green_phases + 1:]]
        return self.radix_encode([phase, elapsed] + density_queue)

    def _discretize_density(self, density):
        if density < 0.1:
            return 0
        elif density < 0.2:
            return 1
        elif density < 0.3:
            return 2
        elif density < 0.4:
            return 3
        elif density < 0.5:
            return 4
        elif density < 0.6:
            return 5
        elif density < 0.7:
            return 6
        elif density < 0.8:
            return 7
        elif density < 0.9:
            return 8
        else:
            return 9

    def _discretize_elapsed_time(self, elapsed):
        elapsed *= self.max_green
        for i in range(self.max_green // self.delta_time):
            if elapsed <= self.delta_time + i * self.delta_time:
                return i
        return self.max_green // self.delta_time - 1

    """
    def radix_encode(self, values):
        res = 0
        for i in range(len(self.radix_factors)):
            res = res * self.radix_factors[i] + values[i]
        return int(res)

    def radix_decode(self, value):
        res = [0 for _ in range(len(self.radix_factors))]
        for i in reversed(range(len(self.radix_factors))):
            res[i] = value % self.radix_factors[i]
            value = value // self.radix_factors[i]
        return res
    """

    def save_csv(self, out_csv_name, step_time):
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            df.to_csv(out_csv_name + '_step{}'.format(step_time) + '.csv', index=False)

    def save_csv2(self, out_csv_name2, step_time):
        if out_csv_name2 is not None:
            df = pd.DataFrame(self.trajectory)
            df.to_csv(out_csv_name2 + '_step{}'.format(step_time) + '.csv', index=False)

    def save_csv3(self, out_csv_name3, step_time):
        if out_csv_name3 is not None:
            df = pd.DataFrame(self.action_info)
            df.to_csv(out_csv_name3 + '_step{}'.format(step_time) + '.csv', index=False)
