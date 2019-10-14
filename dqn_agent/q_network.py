from __future__ import absolute_import, division, print_function

import gin, tensorflow as tf

from tf_agents.networks import encoding_network, network

def validate_specs(action_spec, observation_spec):
    del observation_spec

    flat_action_spec = tf.nest.flatten(action_spec)
    if len(flat_action_spec) > 1:
        raise ValueError("Network only supports action_specs with a single action.")

    if flat_action_spec[0].shape not in [(), (1,)]:
        raise ValueError("Network only supports action_specs with shape in  [(), (1,)])")

@gin.configurable
class QNetwork(network.Network):
    def __init__(self,
                 input_tensor_spec,
                 action_spec,
                 preprocessing_layers=None,
                 preprocessing_combiner=None,
                 conv_layer_prarams=None,
                 fc_layer_params=(75, 40),
                 dropout_layer_params=None,
                 activation_fn=tf.keras.activations.relu,
                 kernel_initializer=None,
                 batch_squash=True,
                 dtype=tf.float32,
                 name="QNetwork"):

        validate_specs(action_spec, input_tensor_spec)
        action_spec = tf.nest.flatten(action_spec)[0]
        num_actions = action_spec.maximum - action_spec.minimum + 1
        encoder_input_tensor_spec = input_tensor_spec

        encoder = encoding_network.EncodingNetwork(encoder_input_tensor_spec,
                                                   preprocessing_layers=preprocessing_layers,
                                                   preprocessing_combiner=preprocessing_combiner,
                                                   conv_layer_params=conv_layer_prarams,
                                                   fc_layer_params=fc_layer_params,
                                                   dropout_layer_params=dropout_layer_params,
                                                   activation_fn=activation_fn,
                                                   kernel_initializer=kernel_initializer,
                                                   batch_squash=batch_squash,
                                                   dtype=dtype)
        q_value_layer = tf.keras.layers.Dense(num_actions,
                                              activation=None,
                                              kernel_initializer=tf.compat.v1.initializers.random_uniform(minval=0.03,
                                                                                                         maxval=0.03),
                                              bias_initializer=tf.compat.v1.initializers.constant(-0.2),
                                              dtype=dtype)
        super(QNetwork, self).__init__(input_tensor_spec=input_tensor_spec,
                                       state_spec=(),
                                       name=name)
        self._encoder = encoder
        self._q_value_layer = q_value_layer

    def call(self, observation, step_type=None, network_state=()):
        state, network_state = self._encoder(observation, step_type=step_type, network_state=network_state)
        return self._q_value_layer(state), network_state