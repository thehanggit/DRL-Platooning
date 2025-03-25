#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 23:23:44 2020

@author: boslavision
"""

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import random
import numpy as np
import scipy as sp
import scipy.io as sio

# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa


def _compute_step_info():
    allVehicles = traci.vehicle.getIDList()
    fuel = 0
    delay = 0
    num_v = 1
    for v in allVehicles:
        fuel += traci.vehicle.getFuelConsumption(v)
        delay += traci.vehicle.getAccumulatedWaitingTime(v)
        num_v += 1
        
    return {
            'avg_wait_time': delay/num_v,
            'avg_fuel_consumption': fuel/num_v
    }
    

def run():
    """execute the TraCI control loop"""
#    sim_time = 0.0
       
    all_delays = []
    all_fuels = []
    Delay_history = []
    Fuel_history = []

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        info = _compute_step_info()
        all_delays.append(info['avg_wait_time'])
        all_fuels.append(info['avg_fuel_consumption'])
            
            
        
    episode_delay = np.mean(all_delays)
    episode_fuel = np.mean(all_fuels)
    print('Average delay: %.2f' %episode_delay)
    print('Average fuel: %.2f' %episode_fuel)
        
    ### Save Data and Save history ###
    Delay_history.append(episode_delay)
    Fuel_history.append(episode_fuel)
    sio.savemat('RL_platoon_result.mat', {'Delay_history': Delay_history, 'Fuel_history':Fuel_history})
    traci.close()
    sys.stdout.flush()












def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


# this is the main entry point of this script
if __name__ == "__main__":
    options = get_options()

    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    traci.start([sumoBinary, "-c", "corridor.sumocfg", "--tripinfo-output", "tripinfo.xml"], label="corridor_test")
    run()