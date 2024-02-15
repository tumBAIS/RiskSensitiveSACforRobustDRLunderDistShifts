"""Generate the training/validation/testing data for the 12 different item distributions"""

import os
import random
import pandas as pd
import numpy as np


vertical_cell_count = 5
horizontal_cell_count = 5
vertical_idx_target = 2
horizontal_idx_target = 0
target_loc = (vertical_idx_target, horizontal_idx_target)
episode_steps = 200
num_episodes = 1000
test_share = 0.1
validation_share = 0.1
base_data_dir = "../data"


### spatial probability distributions
# probabilities are chosen such that the same number of items is generated across all distributions in expectation
# TODO: uncomment the code segment below for the distribution for which data shall be generated

# # gradient 1
# prob_low = 0.24/74
# probs = [[prob_low * (i+1) for i in range(horizontal_cell_count)] for _ in range(vertical_cell_count)]
# data_dir = base_data_dir + f"/data_{vertical_cell_count}x{horizontal_cell_count}_{episode_steps}steps_gradient1"
#
# # gradient 2
# prob_low = 0.24/70
# probs = [[prob_low * (5-i) for i in range(horizontal_cell_count)] for _ in range(vertical_cell_count)]
# data_dir = base_data_dir + f"/data_{vertical_cell_count}x{horizontal_cell_count}_{episode_steps}steps_gradient2"
#
# # gradient 3
# prob_low = 0.24/72
# probs = [[prob_low * (i+1) for _ in range(horizontal_cell_count)] for i in range(vertical_cell_count)]
# data_dir = base_data_dir + f"/data_{vertical_cell_count}x{horizontal_cell_count}_{episode_steps}steps_gradient3"
#
# # gradient 4
# prob_low = 0.24/52
# probs = [[prob_low * 3, prob_low * 2, prob_low, prob_low * 2, prob_low * 3] for _ in range(vertical_cell_count)]
# data_dir = base_data_dir + f"/data_{vertical_cell_count}x{horizontal_cell_count}_{episode_steps}steps_gradient4"
#
# # gradient 5
# prob_low = 0.24/44
# probs = [[prob_low, prob_low * 2, prob_low * 3, prob_low * 2, prob_low] for _ in range(vertical_cell_count)]
# data_dir = base_data_dir + f"/data_{vertical_cell_count}x{horizontal_cell_count}_{episode_steps}steps_gradient5"
#
# # gradient 6
# prob_low = 0.24/54
# probs = [[prob_low * 3 for _ in range(horizontal_cell_count)],
#          [prob_low * 2 for _ in range(horizontal_cell_count)],
#          [prob_low for _ in range(horizontal_cell_count)],
#          [prob_low * 2 for _ in range(horizontal_cell_count)],
#          [prob_low * 3 for _ in range(horizontal_cell_count)]]
# data_dir = base_data_dir + f"/data_{vertical_cell_count}x{horizontal_cell_count}_{episode_steps}steps_gradient6"
#
# # gradient 7
# prob_low = 0.24/42
# probs = [[prob_low for _ in range(horizontal_cell_count)],
#          [prob_low * 2 for _ in range(horizontal_cell_count)],
#          [prob_low * 3 for _ in range(horizontal_cell_count)],
#          [prob_low * 2 for _ in range(horizontal_cell_count)],
#          [prob_low for _ in range(horizontal_cell_count)]]
# data_dir = base_data_dir + f"/data_{vertical_cell_count}x{horizontal_cell_count}_{episode_steps}steps_gradient7"
#
# # gradient 8
# prob_low = 0.24/62
# probs = [[prob_low * 3 for _ in range(horizontal_cell_count)],
#          [prob_low * 3, prob_low * 2, prob_low * 2, prob_low * 2, prob_low * 3],
#          [prob_low * 3, prob_low * 2, prob_low, prob_low * 2, prob_low * 3],
#          [prob_low * 3, prob_low * 2, prob_low * 2, prob_low * 2, prob_low * 3],
#          [prob_low * 3 for _ in range(horizontal_cell_count)]]
# data_dir = base_data_dir + f"/data_{vertical_cell_count}x{horizontal_cell_count}_{episode_steps}steps_gradient8"
#
# # gradient 9
# prob_low = 0.24/34
# probs = [[prob_low for _ in range(horizontal_cell_count)],
#          [prob_low, prob_low * 2, prob_low * 2, prob_low * 2, prob_low],
#          [prob_low, prob_low * 2, prob_low * 3, prob_low * 2, prob_low],
#          [prob_low, prob_low * 2, prob_low * 2, prob_low * 2, prob_low],
#          [prob_low for _ in range(horizontal_cell_count)]]
# data_dir = base_data_dir + f"/data_{vertical_cell_count}x{horizontal_cell_count}_{episode_steps}steps_gradient9"
#
# # uniform
# prob = 0.01
# probs = [[prob for i in range(horizontal_cell_count)] for j in range(vertical_cell_count)]
# data_dir = base_data_dir + f"/data_{vertical_cell_count}x{horizontal_cell_count}_{episode_steps}steps_uniform"
#
# # 3 hot spots: three overlapping distributions with a lot of probability mass in some center location and exponentially
# # decaying probability mass symmetric around the center (decay with increasing Manhattan distance from the center)
# def get_distance(loc1, loc2):
#     (ver_loc1, hor_loc1) = loc1
#     (ver_loc2, hor_loc2) = loc2
#     ver_dist = ver_loc1 - ver_loc2
#     hor_dist = hor_loc1 - hor_loc2
#     return np.abs(ver_dist) + np.abs(hor_dist)
#
# probs = np.zeros((vertical_cell_count, horizontal_cell_count))
# for k in range(vertical_cell_count):
#     for m in range(horizontal_cell_count):
#         for center_loc in [(1,1), (2,3), (4,2)]:
#             probs[k, m] += np.exp(-get_distance((k, m), center_loc))
# probs[target_loc] = 0
# probs /= np.sum(probs)
# probs *= 0.24
# probs = probs.tolist()
# data_dir = base_data_dir + f"/data_{vertical_cell_count}x{horizontal_cell_count}_{episode_steps}steps_hotspots"
#
# # unstructured
# probs = np.zeros((vertical_cell_count, horizontal_cell_count))
# for k in range(vertical_cell_count):
#     for m in range(horizontal_cell_count):
#         probs[k, m] = random.random()
# probs[target_loc] = 0
# probs /= np.sum(probs)
# probs *= 0.24
# probs = probs.tolist()
# data_dir = base_data_dir + f"/data_{vertical_cell_count}x{horizontal_cell_count}_{episode_steps}steps_unstructured"
#
# # unstructured probabilities used to generate the data provided in this repository and used for experiments in the paper
# # 0.01495349 0.00640707 0.00650474 0.01643066 0.0126028
# # 0.00618178 0.00809511 0.00778311 0.00892936 0.01209009
# # 0.         0.01768781 0.00623947 0.01490828 0.00893008
# # 0.01057307 0.01650473 0.01200436 0.01243925 0.00196041
# # 0.0073831  0.00510805 0.00303738 0.01160764 0.01163815

### data generation

# generate directories
episode_data_dir = data_dir + "/episode_data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
if not os.path.exists(episode_data_dir):
    os.makedirs(episode_data_dir)

# generate files defining training/validation/testing split
num_test_episodes = int(num_episodes * test_share)
num_validation_episodes = int(num_episodes * validation_share)
test_episodes = [f"{i:03d}" for i in range(num_test_episodes)]
validation_episodes = [f"{i:03d}" for i in range(num_test_episodes, num_test_episodes + num_validation_episodes)]
training_episodes = [f"{i:03d}" for i in range(num_test_episodes + num_validation_episodes, num_episodes)]
pd.DataFrame({'test_episodes': test_episodes}).to_csv(data_dir + '/test_episodes.csv', index=False)
pd.DataFrame({'validation_episodes': validation_episodes}).to_csv(data_dir + '/validation_episodes.csv', index=False)
pd.DataFrame({'training_episodes': training_episodes}).to_csv(data_dir + '/training_episodes.csv', index=False)

# generate data
for i in range(num_episodes):
    items = []
    for j in range(episode_steps):
        for k in range(vertical_cell_count):
            for m in range(horizontal_cell_count):
                prob = probs[k][m]
                sample = np.random.choice([0, 1], p=[1-prob, prob])
                if sample == 1:
                    if (k, m) != target_loc:
                        items.append([j+1, k, m])
    df = pd.DataFrame(items, columns=["step", "vertical_idx", "horizontal_idx"])
    df.to_csv(episode_data_dir + f"/episode_{i:03d}.csv")
