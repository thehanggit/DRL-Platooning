import gym
import numpy as np

from stable_baselines.deepq import DQN, MlpPolicy

import argparse
import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import pandas as pd
from gym import spaces
import numpy as np
from sumo_rl.environment.env import SumoEnvironment
import traci


if __name__ == '__main__':

    env = SumoEnvironment(net_file='../maps/FourIntersections/test_netconvert.net.xml',
                                    route_file='../maps/FourIntersections/test_netconvert.rou.xml',
                                    out_csv_name='outputs/2way-single-intersection/dqn-vhvh2-stable-mlp-bs',
                                    single_agent=True,
                                    use_gui=True,
                                    num_seconds=100000,
                                    time_to_load_vehicles=120,
                                    max_depart_delay=0,
                                    phases=[
                                        traci.trafficlight.Phase(28, "gGgGgrrrrrgGgGgrrrrrrGrG"),
                                        traci.trafficlight.Phase(5, "gGgGgrrrrrgGgGgrrrrrrrrr"),
                                        traci.trafficlight.Phase(3, "yyyygrrrrryyyygrrrrrrrrr"),
                                        traci.trafficlight.Phase(6, "rrrrGrrrrrrrrrGrrrrrrrrr"),
                                        traci.trafficlight.Phase(3, "rrrryrrrrrrrrryrrrrrrrrr"),
                                        traci.trafficlight.Phase(28, "rrrrrgGgGgrrrrrgGgGgGrGr"),
                                        traci.trafficlight.Phase(5, "rrrrrgGgGgrrrrrgGgGgrrrr"),
                                        traci.trafficlight.Phase(3, "rrrrryyyygrrrrryyyygrrrr"),
                                        traci.trafficlight.Phase(6, "rrrrrrrrrGrrrrrrrrrGrrrr"),
                                        traci.trafficlight.Phase(3, "rrrrrrrrryrrrrrrrrryrrrr")
                                        ])

    model = DQN(
        env=env,
        policy=MlpPolicy,
        learning_rate=1e-3,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02
    )
    model.learn(total_timesteps=100000)