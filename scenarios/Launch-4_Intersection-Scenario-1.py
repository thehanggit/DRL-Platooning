import os
import sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please, declare environment variable 'SUMO_HOME'")

import traci
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from src.simlib import setUpSimulation

setUpSimulation("../maps/FourIntersections/four_traffic_intersection.sumo.cfg",3)
step = 0
while step < 5000:
    traci.simulationStep()
    step += 1

traci.close()
