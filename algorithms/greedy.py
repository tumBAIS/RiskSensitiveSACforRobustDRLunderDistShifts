"""Greedy algorithm"""

import numpy as np
import pandas as pd
from itertools import compress


class Greedy:
    def __init__(self, args, env):
        self.env = env

        self.mode = args.mode
        self.max_episode_steps = args.episode_steps
        self.max_response_time = args.max_response_time
        self.reward = args.reward
        self.data_dir = args.data_dir
        self.results_dir = args.results_dir
        self.validation_episodes = len(
            pd.read_csv(self.data_dir + '/validation_episodes.csv').validation_episodes.tolist())

        with open(self.results_dir + '/args.txt', 'w') as f: f.write(str(args))

    def __call__(self):
        if self.mode == "validation":
            self.validate_policy()
        if self.mode == "testing":
            self.test_policy()

    # compute a greedy action
    def greedy_policy(self):
        # if agent has an item on board, move towards target location
        if self.env.agent_load == 1:
            vertical_delta = self.env.agent_loc[0] - self.env.target_loc[0]
            if vertical_delta != 0:
                if vertical_delta > 0:
                    act = 1
                else:
                    act = 3
            else:
                horizontal_delta = self.env.agent_loc[1] - self.env.target_loc[1]
                if horizontal_delta > 0:
                    act = 4
                else:
                    act = 2

        # if agent has no item on board ...
        else:

            # if there is at least one item available ...
            if self.env.item_locs:
                item_locs_vertical = [i[0] for i in self.env.item_locs]
                item_locs_horizontal = [i[1] for i in self.env.item_locs]
                dist_vertical = [abs(i - self.env.agent_loc[0]) for i in item_locs_vertical]
                dist_horizontal = [abs(i - self.env.agent_loc[1]) for i in item_locs_horizontal]
                dist = np.array(dist_vertical) + np.array(dist_horizontal)
                mask = dist <= self.max_response_time - np.array(self.env.item_times)
                item_locs_vertical = list(compress(item_locs_vertical, mask))
                item_locs_horizontal = list(compress(item_locs_horizontal, mask))

                # if there is at least one item that can be reached by the agent before it disappears ...
                if item_locs_vertical:
                    dist = dist[mask]
                    dist += np.array([abs(i - self.env.target_loc[0]) for i in item_locs_vertical])
                    dist += np.array([abs(i - self.env.target_loc[1]) for i in item_locs_horizontal])
                    profit = self.reward - dist
                    mask = profit > 0
                    item_locs_vertical = list(compress(item_locs_vertical, mask))
                    item_locs_horizontal = list(compress(item_locs_horizontal, mask))

                    # if there is at least one item that will lead to a positive profit ...
                    if item_locs_vertical:
                        profit = profit[mask]
                        idx = np.argmax(profit)
                        item_loc_vertical = item_locs_vertical[idx]
                        item_loc_horizontal = item_locs_horizontal[idx]

                        # move towards the item that will lead to the highest profit
                        vertical_delta = self.env.agent_loc[0] - item_loc_vertical
                        horizontal_delta = self.env.agent_loc[1] - item_loc_horizontal
                        if vertical_delta != 0:
                            if vertical_delta > 0:
                                act = 1
                            else:
                                act = 3
                        elif horizontal_delta != 0:
                            if horizontal_delta > 0:
                                act = 4
                            else:
                                act = 2
                        else:
                            act = 0

                    # otherwise, do not move
                    else:
                        act = 0
                else:
                    act = 0
            else:
                act = 0

        return act

    # compute average reward per validation episode
    def validate_policy(self):
        validation_reward = 0.
        for i in range(self.validation_episodes):
            self.env.reset("validation")
            for j in range(self.max_episode_steps):
                act = self.greedy_policy()
                rew, _, _ = self.env.step(act)
                validation_reward += rew
        avg_validation_reward = validation_reward / self.validation_episodes

        with open(self.results_dir + "/avg_validation_reward.txt", 'w') as f:
            f.write(str(avg_validation_reward.numpy()))

    # compute rewards per test episode
    def test_policy(self):
        test_episodes = pd.read_csv(self.data_dir + '/test_episodes.csv').test_episodes.tolist()

        test_rewards = []
        for i in range(len(test_episodes)):
            test_reward = 0.
            self.env.reset("testing")
            for j in range(self.max_episode_steps):
                act = self.greedy_policy()
                rew, _, _ = self.env.step(act)
                test_reward += rew.numpy()
            test_rewards.append(test_reward)

        df = pd.DataFrame({"test_rewards_greedy": test_rewards}, index=test_episodes)
        df.to_csv(self.results_dir + "/test_rewards.csv")
        with open(self.results_dir + "/avg_test_reward.txt", 'w') as f:
            f.write(str(np.mean(test_rewards)))
