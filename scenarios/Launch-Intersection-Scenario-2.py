import os
import sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please, declare environment variable 'SUMO_HOME'")

import logging
import traci
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from src.simulationmanager import SimulationManager
from src.simlib import setUpSimulation

setUpSimulation("../maps/NormalIntersection/NormalIntersection.sumocfg",3)
step = 0
manager = SimulationManager(True, False)
maxNumAtTrafficLights = 0
while step < 5000:
    manager.handleSimulationStep()
    traci.simulationStep()
    step += 1

logging.info("Max number of stopped cars: %s", manager.maxStoppedVehicles)
logging.info("Average length of platoon: %s", manager.getAverageLengthOfAllPlatoons())
traci.close()
