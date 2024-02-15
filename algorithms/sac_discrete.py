"""Set up networks and define one training iteration for risk-neutral and risk-sensitive discrete SAC"""

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.mixed_precision import LossScaleOptimizer

from neural_network import NN


class SACDiscrete(tf.keras.Model):
    def __init__(self, args):
        super().__init__()

        self.risk_sensitive = tf.constant(args.risk_sensitive)
        self.beta = tf.constant(args.beta)
        self.alpha = tf.Variable(tf.exp(tf.constant(args.log_alpha)), dtype=tf.float32)
        self.tau = tf.constant(args.tau)
        self.huber_delta = tf.constant(args.huber_delta)
        self.gradient_clipping = tf.constant(args.gradient_clipping)
        self.clip_norm = tf.constant(args.clip_norm)
        self.discount = tf.constant(args.discount)

        self.actor = NN(args, name="actor", out_activation="softmax")
        self.qf1 = NN(args, name="qf1", out_activation="linear")
        self.qf2 = NN(args, name="qf2", out_activation="linear")
        self.qf1_target = NN(args, name="qf1_target", out_activation="linear")
        self.qf2_target = NN(args, name="qf2_target", out_activation="linear")

        dummy_obs = tf.ones((1, args.vertical_cell_count, args.horizontal_cell_count, 3))
        self.qf1(dummy_obs)
        self.qf2(dummy_obs)
        self.qf1_target(dummy_obs)
        self.qf2_target(dummy_obs)
        for target_var, source_var in zip(self.qf1_target.weights, self.qf1.weights):
            target_var.assign(source_var)
        for target_var, source_var in zip(self.qf2_target.weights, self.qf2.weights):
            target_var.assign(source_var)

        lr = args.lr
        self.actor_optimizer = LossScaleOptimizer(Adam(lr))
        self.qf1_optimizer = LossScaleOptimizer(Adam(lr))
        self.qf2_optimizer = LossScaleOptimizer(Adam(lr))

        self.q1_update = tf.function(self.q_update)
        self.q2_update = tf.function(self.q_update)

    # get action for obs input without batch dim
    @tf.function
    def get_action(self, obs, test=tf.constant(False)):
        obs = tf.expand_dims(obs, axis=0)
        probs = self.actor(obs)
        probs = tf.squeeze(probs, axis=[0])

        if test:
            return tf.argmax(probs, output_type=tf.int32)
        else:
            return tfp.distributions.Categorical(probs=probs).sample()

    # define one training iteration for a batch of experience
    def train(self, obses, acts, rews, next_obses, dones):
        q_loss, policy_loss, mean_ent = self.train_body(obses, acts, rews, next_obses, dones)

        tf.summary.scalar(name="critic_loss", data=q_loss)
        tf.summary.scalar(name="actor_loss", data=policy_loss)
        tf.summary.scalar(name="mean_ent", data=mean_ent)

    @tf.function
    def train_body(self, obses, acts, rews, next_obses, dones):
        if self.risk_sensitive:
            target_q = self.target_Qs_risk(rews, next_obses, dones)
        else:
            target_q = self.target_Qs(rews, next_obses, dones)

        q1_loss, cur_q1 = self.q1_update(obses, acts, target_q, self.qf1, self.qf1_optimizer, self.qf1_target)
        q2_loss, cur_q2 = self.q2_update(obses, acts, target_q, self.qf2, self.qf2_optimizer, self.qf2_target)
        q_loss = (q1_loss + q2_loss) / 2.

        policy_loss, cur_act_prob, cur_act_logp = self.actor_update(obses, cur_q1, cur_q2)

        # mean entropy (info for summary output, not needed for algorithm)
        mean_ent = tf.reduce_mean(-tf.einsum('ij,ij->i', cur_act_prob, cur_act_logp))

        return q_loss, policy_loss, mean_ent

    # target Qs for the risk-neutral case
    @tf.function
    def target_Qs(self, rews, next_obses, dones):
        next_act_prob = self.actor(next_obses)
        next_act_logp = tf.math.log(next_act_prob + 1e-8)

        next_q1_target = self.qf1_target(next_obses)
        next_q2_target = self.qf2_target(next_obses)
        next_q = tf.minimum(next_q1_target, next_q2_target)

        target_q = tf.einsum('ij,ij->i', next_act_prob, next_q - self.alpha * next_act_logp)

        return rews + (1. - tf.cast(dones, tf.float32)) * self.discount * target_q

    # target Qs for the risk-sensitive case
    @tf.function
    def target_Qs_risk(self, rews, next_obses, dones):
        next_act_prob = self.actor(next_obses)
        next_act_logp = tf.math.log(next_act_prob + 1e-8)
        ent = -tf.einsum('ij,ij->i', next_act_prob, next_act_logp)

        next_q1_target = self.qf1_target(next_obses)
        next_q2_target = self.qf2_target(next_obses)
        next_q = tf.minimum(next_q1_target, next_q2_target)
        max_next_q = tf.reduce_max(next_q, axis=1)

        exp_term = tf.exp(self.beta * self.discount * (next_q - tf.expand_dims(max_next_q, axis=1)))
        exp_term = tf.math.log(tf.einsum('ij,ij->i', next_act_prob, exp_term) + 1e-8) / self.beta

        return rews + (1. - tf.cast(dones, tf.float32)) * (self.discount * (self.alpha * ent + max_next_q) + exp_term)

    # update (target) critics
    def q_update(self, obses, acts, target_q, qf, qf_optimizer, qf_target):
        with tf.GradientTape() as tape:
            cur_q = qf(obses)  # gives Q(s) for all a, not Q(s,a) for one a
            cur_q_selected = tf.gather_nd(cur_q, tf.expand_dims(acts, axis=1), batch_dims=1)  # get Q(s,a) from Q(s)

            q_loss = self.huber_loss(target_q - cur_q_selected, self.huber_delta)
            q_loss = tf.reduce_mean(q_loss)

            regularization_loss = tf.reduce_sum(qf.losses)
            scaled_q_loss = qf_optimizer.get_scaled_loss(q_loss + regularization_loss)

        scaled_gradients = tape.gradient(scaled_q_loss, qf.trainable_weights)
        gradients = qf_optimizer.get_unscaled_gradients(scaled_gradients)
        if self.gradient_clipping:
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)
        qf_optimizer.apply_gradients(zip(gradients, qf.trainable_weights))

        for target_var, source_var in zip(qf_target.weights, qf.weights):
            target_var.assign(self.tau * source_var + (1. - self.tau) * target_var)

        return q_loss, cur_q

    # Huber loss: MSE for -delta < x < delta, linear otherwise
    @tf.function
    def huber_loss(self, x, delta):
        delta = tf.ones_like(x) * delta
        less_than_max = 0.5 * tf.square(x)  # MSE
        greater_than_max = delta * (tf.abs(x) - 0.5 * delta)  # linear
        return tf.where(tf.abs(x) <= delta, less_than_max, greater_than_max)

    # update the actor
    @tf.function
    def actor_update(self, obses, cur_q1, cur_q2):
        with tf.GradientTape() as tape:
            cur_act_prob = self.actor(obses)
            cur_act_logp = tf.math.log(cur_act_prob + 1e-8)

            policy_loss = tf.einsum('ij,ij->i', cur_act_prob, self.alpha * cur_act_logp - tf.minimum(cur_q1, cur_q2))
            policy_loss = tf.reduce_mean(policy_loss)

            regularization_loss = tf.reduce_sum(self.actor.losses)
            scaled_loss = self.actor_optimizer.get_scaled_loss(policy_loss + regularization_loss)

        scaled_gradients = tape.gradient(scaled_loss, self.actor.trainable_weights)
        gradients = self.actor_optimizer.get_unscaled_gradients(scaled_gradients)
        if self.gradient_clipping:
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)
        self.actor_optimizer.apply_gradients(zip(gradients, self.actor.trainable_weights))

        return policy_loss, cur_act_prob, cur_act_logp
