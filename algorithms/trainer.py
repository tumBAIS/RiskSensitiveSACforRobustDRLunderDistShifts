"""Training loop incl. validation and testing"""

import os
import copy
import pandas as pd
import numpy as np
import tensorflow as tf

from replay_buffer import ReplayBuffer


class Trainer:
    def __init__(self, args, policy, env):
        self.policy = policy
        self.env = env

        self.vertical_cell_count = args.vertical_cell_count
        self.horizontal_cell_count = args.horizontal_cell_count
        self.max_episode_steps = args.episode_steps
        self.max_steps = args.max_steps
        self.min_steps = args.min_steps
        self.random_steps = args.random_steps
        self.update_interval = args.update_interval
        self.validation_interval = args.validation_interval
        self.tracking_interval = args.tracking_interval
        self.rb_size = args.rb_size
        self.batch_size = args.batch_size
        self.scheduled_alpha = args.scheduled_alpha
        self.alpha_schedule_steps = args.alpha_schedule_steps
        self.alpha_schedule_value = args.alpha_schedule_value
        self.data_dir = args.data_dir
        self.results_dir = args.results_dir
        self.validation_episodes = len(
            pd.read_csv(self.data_dir + '/validation_episodes.csv').validation_episodes.tolist())

        # save arguments and environment variables
        with open(self.results_dir + '/args.txt', 'w') as f: f.write(str(args))
        with open(self.results_dir + '/environ.txt', 'w') as f: f.write(str(dict(os.environ)))

        # initialize model saving and potentially restore saved model
        self.set_checkpoint(args.model_dir)

        # prepare TensorBoard output
        self.writer = tf.summary.create_file_writer(self.results_dir)
        self.writer.set_as_default()

    # initialize model saving and potentially restore saved model
    def set_checkpoint(self, model_dir):
        self.checkpoint = tf.train.Checkpoint(self.policy)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, directory=self.results_dir,
                                                             max_to_keep=1)

        if model_dir is not None:
            assert os.path.isdir(model_dir)
            latest_path_ckpt = tf.train.latest_checkpoint(model_dir)
            self.checkpoint.restore(latest_path_ckpt)

    def __call__(self):
        total_steps = 0
        episode_steps = 0
        episode_reward = 0.
        self.best_validation_reward = -1000000  # some very small number

        replay_buffer = ReplayBuffer(self.rb_size, self.vertical_cell_count, self.horizontal_cell_count)

        obs = self.env.reset()

        while total_steps < self.max_steps:
            if (total_steps + 1) % 100 == 0: tf.print("Started step", total_steps + 1)

            if self.scheduled_alpha:
                if total_steps == self.alpha_schedule_steps:
                    self.policy.alpha.assign(self.alpha_schedule_value)

            if total_steps < self.random_steps:
                act = tf.random.uniform(shape=[], maxval=5, dtype=tf.int32)
            else:
                act = self.policy.get_action(obs)

            rew, next_obs, done = self.env.step(act)

            replay_buffer.add(obs, act, rew, next_obs, done)

            obs = next_obs
            total_steps += 1
            episode_steps += 1
            episode_reward += rew

            tf.summary.experimental.set_step(total_steps)

            if total_steps >= self.min_steps and total_steps % self.update_interval == 0:
                obses, acts, rews, next_obses, dones = replay_buffer.sample(self.batch_size)
                with tf.summary.record_if(total_steps % self.tracking_interval == 0):
                    self.policy.train(obses, acts, rews, next_obses, dones)

            if total_steps % self.validation_interval == 0:
                avg_validation_reward = self.validate_policy()
                tf.summary.scalar(name="avg_reward_per_validation_episode", data=avg_validation_reward)

                # save model only if it performs better than the best one tested on the validation data so far and
                # if a scheduled alpha is used, only after 100k steps have elapsed after alpha was set to final value
                condition = False
                if self.scheduled_alpha:
                    if total_steps >= self.alpha_schedule_steps + 100000:
                        condition = True
                else:
                    condition = True
                if condition and (avg_validation_reward > self.best_validation_reward):
                    self.best_validation_reward = avg_validation_reward
                    self.checkpoint_manager.save()

            if episode_steps == self.max_episode_steps:
                tf.summary.scalar(name="training_reward", data=episode_reward)
                episode_steps = 0
                episode_reward = 0.
                obs = self.env.reset()

        tf.summary.flush()

        self.test_policy()

        tf.print("Finished")

    # compute average reward per validation episode achieved by current policy
    def validate_policy(self):
        validation_reward = 0.
        for i in range(self.validation_episodes):
            obs = self.env.reset("validation")
            for j in range(self.max_episode_steps):
                act = self.policy.get_action(obs, test=tf.constant(True))
                rew, next_obs, _ = self.env.step(act)
                validation_reward += rew
                obs = next_obs
        avg_validation_reward = validation_reward / self.validation_episodes

        # reset list of remaining validation episodes
        self.env.remaining_validation_episodes = copy.deepcopy(self.env.validation_episodes)

        return avg_validation_reward

    # compute rewards per test episode with best policy
    def test_policy(self):
        latest_path_ckpt = tf.train.latest_checkpoint(self.results_dir)
        self.checkpoint.restore(latest_path_ckpt)

        test_episodes = pd.read_csv(self.data_dir + '/test_episodes.csv').test_episodes.tolist()

        test_rewards = []
        for i in range(len(test_episodes)):
            test_reward = 0.
            obs = self.env.reset("testing")
            for j in range(self.max_episode_steps):
                act = self.policy.get_action(obs, test=tf.constant(True))
                rew, next_obs, _ = self.env.step(act)
                test_reward += rew.numpy()
                obs = next_obs
            test_rewards.append(test_reward)

        df = pd.DataFrame({"test_rewards_RL": test_rewards}, index=test_episodes)
        df.to_csv(self.results_dir + "/test_rewards.csv")
        with open(self.results_dir + "/avg_test_reward.txt", 'w') as f:
            f.write(str(np.mean(test_rewards)))
