"""Neural network for actor and (target) critics. When used as critic, computes Q-values for all possible actions."""

import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Activation, Conv2D
from tensorflow.keras.regularizers import L2


class NN(tf.keras.Model):
    def __init__(self, args, name, out_activation):
        super().__init__(name=name)

        reg_coef = args.regularization_coefficient

        self.conv_layer1 = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform',
                                  kernel_regularizer=L2(reg_coef))
        self.conv_layer2 = Conv2D(64, (2, 2), padding='same', activation='relu', kernel_initializer='he_uniform',
                                  kernel_regularizer=L2(reg_coef))
        self.conv_layer3 = Conv2D(64, (2, 2), padding='same', activation='relu', kernel_initializer='he_uniform',
                                  kernel_regularizer=L2(reg_coef))

        self.flatten_layer = Flatten()

        self.dense_layer1 = Dense(256, activation="relu", kernel_initializer='he_uniform',
                                  kernel_regularizer=L2(reg_coef))
        self.dense_layer2 = Dense(256, activation="relu", kernel_initializer='he_uniform',
                                  kernel_regularizer=L2(reg_coef))

        self.output_layer = Dense(5, activation=None, kernel_initializer='glorot_uniform',
                                  kernel_regularizer=L2(reg_coef))
        self.output_activation = Activation(out_activation, dtype='float32')

    @tf.function
    def call(self, obs):
        features = self.conv_layer1(obs)
        features = self.conv_layer2(features)
        features = self.conv_layer3(features)
        features = self.flatten_layer(features)
        features = self.dense_layer1(features)
        features = self.dense_layer2(features)
        return self.output_activation(self.output_layer(features))
