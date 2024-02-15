"""Replace item locations by items from uniform distribution with a certain probability and save resulting episode data
to a new directory"""

import os
import pandas as pd
import random

vertical_cell_count = 5
horizontal_cell_count = 5
episode_steps = 200
num_episodes = 1000
data_dir = "../data"

original_data_dir = data_dir + f"/data_{vertical_cell_count}x{horizontal_cell_count}_{episode_steps}steps_gradient1/episode_data"

# possible item locations
lookup_list = [(0,0), (0,1), (0,2), (0,3), (0,4),
               (1,0), (1,1), (1,2), (1,3), (1,4),
                      (2,1), (2,2), (2,3), (2,4),
               (3,0), (3,1), (3,2), (3,3), (3,4),
               (4,0), (4,1), (4,2), (4,3), (4,4)]

# probabilities that an item location is replaced by an item location drawn from a uniform distribution
shares = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# random seeds to repeat the data adaptation len(seeds) times
# to get multiple adapted data sets over which results can be averaged
seeds = [0, 1, 2]

for share in shares:
    for seed in seeds:
        new_data_dir = data_dir + f"/data_{vertical_cell_count}x{horizontal_cell_count}_{episode_steps}steps_gradient1_{int(share*100)}uniform_{seed}"
        new_episode_data_dir = new_data_dir + "/episode_data"

        if not os.path.exists(new_data_dir):
            os.makedirs(new_data_dir)
        if not os.path.exists(new_episode_data_dir):
            os.makedirs(new_episode_data_dir)

        for episode in range(num_episodes):
            data = pd.read_csv(original_data_dir + f"/episode_{episode:03d}.csv", index_col=0)
            for i in range(len(data.index)):
                if random.random() < share:
                    new_item_loc_idx = random.randint(0, 23)
                    (new_item_loc_idx_vertical, new_item_loc_idx_horizontal) = lookup_list[new_item_loc_idx]
                    data.loc[i, "vertical_idx"] = new_item_loc_idx_vertical
                    data.loc[i, "horizontal_idx"] = new_item_loc_idx_horizontal
            data.to_csv(new_episode_data_dir + f"/episode_{episode:03d}.csv")
