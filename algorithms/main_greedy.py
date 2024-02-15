"""Parse parameters and run greedy algorithm"""

import argparse

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

parser.add_argument('--mode', type=str)  # validation or testing
parser.add_argument('--vertical_cell_count', type=int)  # number of vertical cells in grid
parser.add_argument('--horizontal_cell_count', type=int)  # number of horizontal cells in grid
parser.add_argument('--vertical_idx_target', type=int)  # vertical index of target location
parser.add_argument('--horizontal_idx_target', type=int)  # horizontal index of target location
parser.add_argument('--episode_steps', type=int)  # number of steps per episode
parser.add_argument('--max_response_time', type=int)  # number of steps until items disappear if not picked up
parser.add_argument('--reward', type=float)  # positive reward obtained when item is delivered to target location
parser.add_argument('--data_dir', type=str)  # relative path to directory where data is stored
parser.add_argument('--results_dir', type=str)  # relative path to directory where results shall be saved

args = parser.parse_args()

from environment import Environment
from greedy import Greedy

env = Environment(args)
greedy = Greedy(args, env)

greedy()
