import logging
import traci
"""
from sumolib import checkBinary
"""

def flatten(l):
    # A basic function to flatten a list
    return [item for sublist in l for item in sublist]

def setUpSimulation(configFile, sumoBinary, trafficScale = 1):
    """
    # Check SUMO has been set up properly
    sumoBinary = checkBinary("sumo-gui")
    """
    
    # Set up logger
    logging.basicConfig(format='%(asctime)s %(message)s')
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    """
    # Start Simulation and step through
    traci.start([sumoBinary, 
                 "-c", configFile, 
                 "--step-length", "1", 
                 "--collision.action", "none", 
                 "--start",
                 "--additional-files", "../outputs/additional.xml", 
                 "--duration-log.statistics", 
                 "--scale", str(trafficScale),
                 "--log", "../outputs/RL_platoon/sumo_step_info.txt",
                 "--no-step-log",
                 "--no-warnings"])