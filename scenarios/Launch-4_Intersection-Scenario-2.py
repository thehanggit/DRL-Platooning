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
from platoon_src.simulationmanager import SimulationManager
from platoon_src.simlib import setUpSimulation

setUpSimulation("../maps/FourIntersections/four_traffic_intersection.sumo.cfg",3)
step = 0
manager = SimulationManager(10, 1, True, False)
maxNumAtTrafficLights = 0
#traci.gui.setZoom(traci.gui.DEFAULT_VIEW, 150)
while step < 5000:
    manager.handleSimulationStep()
    traci.simulationStep()
    #traci.gui.screenshot(traci.gui.DEFAULT_VIEW , '../screenshots/img_' + str(step) + '.png')
    step += 1

logging.info("Max number of stopped cars: %s", manager.maxStoppedVehicles)
logging.info("Average length of platoon: %s", manager.getAverageLengthOfAllPlatoons())
traci.close()
