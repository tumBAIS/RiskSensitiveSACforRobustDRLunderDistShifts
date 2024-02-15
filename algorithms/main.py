"""Parse parameters and run algorithm"""

# parse parameters
import argparse

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

parser.add_argument('--risk_sensitive', type=str)  # if the risk-sensitive algorithm is used (risk-neutral if FALSE)
parser.add_argument('--vertical_cell_count', type=int)  # no. of vertical cells in grid
parser.add_argument('--horizontal_cell_count', type=int)  # no. of horizontal cells in grid
parser.add_argument('--vertical_idx_target', type=int)  # vertical index of target location
parser.add_argument('--horizontal_idx_target', type=int)  # horizontal index of target location
parser.add_argument('--episode_steps', type=int)  # no. of steps per episode
parser.add_argument('--max_response_time', type=int)  # no. of steps until items disappear if not picked up
parser.add_argument('--reward', type=float)  # positive reward obtained when item is delivered to target location
parser.add_argument('--random_seed', type=int)  # random seed
parser.add_argument('--max_steps', type=int)  # no. of steps to interact with environment
parser.add_argument('--min_steps', type=int)  # no. of steps before neural net weight updates begin
parser.add_argument('--random_steps', type=int)  # no. of steps with random policy at the beginning
parser.add_argument('--update_interval', type=int)  # no. of steps between neural net weight updates
parser.add_argument('--validation_interval', type=int)  # no. of steps between validation runs (must be multiple of no. of time steps per episode)
parser.add_argument('--tracking_interval', type=int)  # interval at which training data is saved
parser.add_argument('--regularization_coefficient', type=float)  # coefficient for L2 regularization of networks (0 if no regularization)
parser.add_argument('--rb_size', type=int)  # replay buffer size
parser.add_argument('--batch_size', type=int)  # (mini-)batch size
parser.add_argument('--beta', type=float)  # beta value that controls the risk-sensitivity
parser.add_argument('--log_alpha', type=float)  # log(alpha), where alpha is the entropy coefficient
parser.add_argument('--scheduled_alpha', type=str)  # whether alpha follows a schedule
parser.add_argument('--alpha_schedule_steps', type=int)  # steps after which alpha is set to alpha_schedule_value if alpha is scheduled
parser.add_argument('--alpha_schedule_value', type=float)  # value to which alpha (not log alpha!) is set after alpha_schedule_steps if alpha is scheduled
parser.add_argument('--tau', type=float)  # smoothing factor for exponential moving average to update target critic parameters
parser.add_argument('--huber_delta', type=float)  # delta value at which Huber loss becomes linear
parser.add_argument('--gradient_clipping', type=str)  # whether gradient clipping is applied
parser.add_argument('--clip_norm', type=float)  # global norm used for gradient clipping
parser.add_argument('--lr', type=float)  # learning rate
parser.add_argument('--discount', type=float)  # discount factor
parser.add_argument('--data_dir', type=str)  # relative path to directory where data is stored
parser.add_argument('--results_dir', type=str)  # relative path to directory where results shall be saved
parser.add_argument('--model_dir', type=str, default=None)  # relative path to directory with saved model that shall be restored in the beginning (overwriting default initialization of network weights)

args = parser.parse_args()

if args.risk_sensitive == "False":
    args.risk_sensitive = False
elif args.risk_sensitive == "True":
    args.risk_sensitive = True
else:
    raise argparse.ArgumentTypeError('True or False expected for argument --risk_sensitive.')

if args.scheduled_alpha == "False":
    args.scheduled_alpha = False
elif args.scheduled_alpha == "True":
    args.scheduled_alpha = True
else:
    raise argparse.ArgumentTypeError('True or False expected for argument --scheduled_alpha.')

if args.gradient_clipping == "False":
    args.gradient_clipping = False
elif args.gradient_clipping == "True":
    args.gradient_clipping = True
else:
    raise argparse.ArgumentTypeError('True or False expected for argument --gradient_clipping.')

# set seed and further global settings
seed = args.random_seed

import os
os.environ['PYTHONHASHSEED'] = str(seed)

os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=2"  # enable XLA

import random
random.seed(seed)

import numpy as np
np.random.seed(seed)

import tensorflow as tf
tf.random.set_seed(seed)

tf.keras.mixed_precision.set_global_policy('mixed_float16')  # enable mixed precision computations

# initialize environment, RL algorithm and trainer
from environment import Environment
from sac_discrete import SACDiscrete
from trainer import Trainer

env = Environment(args)
policy = SACDiscrete(args)
trainer = Trainer(args, policy, env)

# call trainer
trainer()
