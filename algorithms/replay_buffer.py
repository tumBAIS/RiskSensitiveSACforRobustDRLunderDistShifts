"""Replay buffer"""

import tensorflow as tf


class ReplayBuffer(object):
    def __init__(self, size, vertical_cell_count, horizontal_cell_count):
        self.obses = tf.Variable(tf.zeros((size, vertical_cell_count, horizontal_cell_count, 3,), dtype=tf.float32))
        self.acts = tf.Variable(tf.zeros((size,), dtype=tf.int32))
        self.rews = tf.Variable(tf.zeros((size,), dtype=tf.float32))
        self.next_obses = tf.Variable(tf.zeros((size, vertical_cell_count, horizontal_cell_count, 3,),
                                               dtype=tf.float32))
        self.dones = tf.Variable(tf.zeros((size,), dtype=tf.int32))

        self.maxsize = size
        self.size = tf.Variable(0, dtype=tf.int32)
        self.next_idx = tf.Variable(0, dtype=tf.int32)

    @tf.function
    def add(self, obs, act, rew, next_obs, done):
        self.obses.scatter_nd_update([[self.next_idx]], [obs])
        self.acts.scatter_nd_update([[self.next_idx]], [act])
        self.rews.scatter_nd_update([[self.next_idx]], [rew])
        self.next_obses.scatter_nd_update([[self.next_idx]], [next_obs])
        self.dones.scatter_nd_update([[self.next_idx]], [done])

        self.size.assign(tf.math.minimum(self.size + 1, self.maxsize))
        self.next_idx.assign((self.next_idx + 1) % self.maxsize)

    @tf.function
    def sample(self, batch_size):
        idxes = tf.random.uniform((batch_size,), maxval=self.size, dtype=tf.int32)

        obses = tf.gather(self.obses, idxes)
        acts = tf.gather(self.acts, idxes)
        rews = tf.gather(self.rews, idxes)
        next_obses = tf.gather(self.next_obses, idxes)
        dones = tf.gather(self.dones, idxes)

        rews /= tf.math.reduce_std(self.rews[:self.size])

        return obses, acts, rews, next_obses, dones
