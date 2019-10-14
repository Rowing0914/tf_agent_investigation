from __future__ import absolute_import, division, print_function

import gin
import tensorflow as tf

from tf_agents.networks import network

@gin.configurable
class Network(network.Network):
    def __init__(self,
                 input_tensor_spec,
                 action_spec,
                 dtype=tf.float32,
                 name="Network"):
        action_spec = tf.nest.flatten(action_spec)[0]
        num_actions = action_spec.maximum - action_spec.minimum + 1
        super(Network, self).__init__(input_tensor_spec=input_tensor_spec,
                                      state_spec=(),
                                      name=name)

        self._dense1 = tf.keras.layers.Dense(100,
                                             activation=None,
                                             kernel_initializer=tf.compat.v1.initializers.random_uniform(minval=-0.03,
                                                                                                         maxval=0.03),
                                             bias_initializer=tf.compat.v1.initializers.constant(-0.2),
                                             dtype=dtype)
        self._dense2 = tf.keras.layers.Dense(50,
                                             activation=None,
                                             kernel_initializer=tf.compat.v1.initializers.random_uniform(minval=-0.03,
                                                                                                         maxval=0.03),
                                             bias_initializer=tf.compat.v1.initializers.constant(-0.2),
                                             dtype=dtype)
        self._dense3 = tf.keras.layers.Dense(num_actions,
                                             activation=None,
                                             kernel_initializer=tf.compat.v1.initializers.random_uniform(minval=-0.03,
                                                                                                         maxval=0.03),
                                             bias_initializer=tf.compat.v1.initializers.constant(-0.2),
                                             dtype=dtype)


    def call(self, observation, step_type=None, network_state=()):
        x = self._dense1(observation)
        x = self._dense2(x)
        out = self._dense3(x)
        return out, network_state