import argparse
import os
import sys
import pandas as pd

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from sumo_rl_src.environment.env import SumoEnvironment
from sumo_rl_src.agents.ql_agent import QLAgent
from sumo_rl_src.exploration.epsilon_greedy import EpsilonGreedy


if __name__ == '__main__':

    alpha = 0.1
    gamma = 0.99
    decay = 1
    runs = 1

    env = SumoEnvironment(net_file='../maps/FourIntersections/test_netconvert.net.xml',
                          route_file='../maps/FourIntersections/test_netconvert.rou.xml',
                          use_gui=True,
                          num_seconds=40000,
                          time_to_load_vehicles=300,
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

    for run in range(1, runs+1):
        initial_states = env.reset()
        ql_agents = {ts: QLAgent(starting_state=env.encode(initial_states[ts]),
                                 state_space=env.observation_space,
                                 action_space=env.action_space,
                                 alpha=alpha,
                                 gamma=gamma,
                                 exploration_strategy=EpsilonGreedy(initial_epsilon=0.05, min_epsilon=0.005, decay=decay)) for ts in env.ts_ids}
        infos = []
        done = {'__all__': False}
        #count = 0
        #traci.gui.setZoom(traci.gui.DEFAULT_VIEW, 150)
        while not done['__all__']:
            #traci.gui.screenshot(traci.gui.DEFAULT_VIEW, '../screenshots/img_' + str(count) + '.png')
            #count += 1
            
            actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

            s, r, done, info = env.step(action=actions)
            infos.append(info)

            for agent_id in ql_agents.keys():
                ql_agents[agent_id].learn(next_state=env.encode(s[agent_id]), reward=r[agent_id])

        env.close()

        df = pd.DataFrame(infos)
        df.to_csv('outputs/4x4grid/c2_alpha{}_gamma{}_decay{}_run{}.csv'.format(alpha, gamma, decay, run), index=False)

