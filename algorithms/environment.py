"""Environment: episode initialization, transition and reward function to get next states and rewards.
Gives state encodings as needed for the neural networks.

Positions in the grid are encoded as follows:
- (0,0) is upper left corner
- first index is vertical (increasing from top to bottom)
- second index is horizontal (increasing from left to right)

Actions are encoded as follows: 0 (do not move), 1 (move up), 2 (move right), 3 (move down), 4 (move left)

If a new item appears in a cell into which the agent moves/at which the agent stays in the same time step,
it is not picked up (if agent wants to pick it up, it has to stay in the cell in the next time step)"""

import random
import numpy as np
import pandas as pd
import tensorflow as tf
from copy import deepcopy
from itertools import compress


class Environment(object):
    def __init__(self, args):
        self.vertical_cell_count = args.vertical_cell_count
        self.horizontal_cell_count = args.horizontal_cell_count
        self.vertical_idx_target = args.vertical_idx_target
        self.horizontal_idx_target = args.horizontal_idx_target
        self.target_loc = (self.vertical_idx_target, self.horizontal_idx_target)
        self.episode_steps = args.episode_steps
        self.max_response_time = args.max_response_time
        self.reward = args.reward
        self.data_dir = args.data_dir

        self.training_episodes = pd.read_csv(self.data_dir + '/training_episodes.csv').training_episodes.tolist()
        self.validation_episodes = pd.read_csv(self.data_dir + '/validation_episodes.csv').validation_episodes.tolist()
        self.test_episodes = pd.read_csv(self.data_dir + '/test_episodes.csv').test_episodes.tolist()
        self.remaining_training_episodes = deepcopy(self.training_episodes)
        self.remaining_validation_episodes = deepcopy(self.validation_episodes)

    # initialize new episode
    def reset(self, mode="training"):
        modes = ['training', 'validation', 'testing']
        if mode not in modes:
            raise ValueError("Invalid mode. Expected one of: %s" % modes)

        self.step_count = 0  # time step
        self.agent_loc = (self.vertical_idx_target, self.horizontal_idx_target)  # agent location
        self.agent_load = 0  # can be 0 (no item loaded) or 1 (item loaded)
        self.item_locs = []  # locations of items available (not picked up, not disappeared)
        self.item_times = []  # how many time steps elapsed since item appeared

        # pick episode and prepare item data for this episode
        if mode == "testing":
            self.episode = self.test_episodes[0]
            self.test_episodes.remove(self.episode)
        elif mode == "validation":
            self.episode = self.remaining_validation_episodes[0]
            self.remaining_validation_episodes.remove(self.episode)
        else:
            if not self.remaining_training_episodes:
                self.remaining_training_episodes = deepcopy(self.training_episodes)
            self.episode = random.choice(self.remaining_training_episodes)
            self.remaining_training_episodes.remove(self.episode)
        self.data = pd.read_csv(self.data_dir + f'/episode_data/episode_{self.episode:03d}.csv', index_col=0)

        return self.get_obs()

    # transition to next state and corresponding reward computation
    def step(self, act):
        self.step_count += 1
        if self.step_count == self.episode_steps:
            done = 1
        else:
            done = 0

        # update agent location based on move and incur cost for the move
        # (if action results in agent leaving the grid, it does not move)
        rew = 0
        if act == 1:  # up action
            if self.agent_loc[0] > 0:
                self.agent_loc = (self.agent_loc[0] - 1, self.agent_loc[1])
                rew += -1
        elif act == 2:  # right action
            if self.agent_loc[1] < self.horizontal_cell_count - 1:
                self.agent_loc = (self.agent_loc[0], self.agent_loc[1] + 1)
                rew += -1
        elif act == 3:  # down action
            if self.agent_loc[0] < self.vertical_cell_count - 1:
                self.agent_loc = (self.agent_loc[0] + 1, self.agent_loc[1])
                rew += -1
        elif act == 4:  # left action
            if self.agent_loc[1] > 0:
                self.agent_loc = (self.agent_loc[0], self.agent_loc[1] - 1)
                rew += -1

        # pick up item if no item already on board and item available in cell to which agent just moved
        # 1/2 of the reward for the item obtained at pick-up
        if self.agent_load == 0:
            if self.agent_loc in self.item_locs:
                self.agent_load = 1
                idx = self.item_locs.index(self.agent_loc)
                self.item_locs.pop(idx)
                self.item_times.pop(idx)
                rew += self.reward / 2

        # drop off item if target location reached, 1/2 of the reward for the item obtained at drop-off
        elif self.agent_loc == self.target_loc:
            self.agent_load = 0
            rew += self.reward / 2

        # increase time counter for all items that were not picked up
        self.item_times = [i + 1 for i in self.item_times]

        # remove items for which max response time is reached
        mask = [i < self.max_response_time for i in self.item_times]
        self.item_locs = list(compress(self.item_locs, mask))
        self.item_times = list(compress(self.item_times, mask))

        # generate new items that appear in the current time step
        new_items = self.data[self.data.step == self.step_count]
        new_items = list(zip(new_items.vertical_idx, new_items.horizontal_idx))
        new_items = [i for i in new_items if i not in self.item_locs]  # not more than one item per cell
        self.item_locs += new_items
        self.item_times += [0] * len(new_items)

        next_obs = self.get_obs()

        return tf.constant(rew, tf.float32), next_obs, tf.constant(done, tf.int32)

    # get state encoding (observation) as needed for the neural networks based on the current system state
    def get_obs(self):
        target_channel = np.zeros((self.vertical_cell_count, self.horizontal_cell_count))
        target_channel[self.target_loc] = 1

        items_channel = np.zeros((self.vertical_cell_count, self.horizontal_cell_count))
        for (i, item_loc) in enumerate(self.item_locs):
            items_channel[item_loc] = (self.max_response_time - self.item_times[i]) / self.max_response_time

        agent_channel = np.zeros((self.vertical_cell_count, self.horizontal_cell_count))
        agent_channel[self.agent_loc] = 1
        if self.agent_load == 1:
            agent_channel /= 2

        return tf.stack([target_channel, items_channel, agent_channel], axis=-1)
