from __future__ import absolute_import, division, print_function

import gin
import tensorflow as tf
from tf_agents.policies import tf_policy
from tf_agents.distributions import shifted_categorical
from tf_agents.trajectories import policy_step

@gin.configurable
class Policy(tf_policy.Base):
    def __init__(self,
                 time_step_spec,
                 action_spec,
                 network,
                 name=None):
        self._flat_action_spec = tf.nest.flatten(action_spec)[0]
        self._network = network
        super(Policy, self).__init__(time_step_spec,
                                     action_spec,
                                     policy_state_spec=network.state_spec,
                                     clip=False,
                                     emit_log_probability=False,
                                     name=name)

    def _variables(self):
        return self._network.variables

    def _distribution(self, time_step, policy_state):
        network_obs = time_step.observation

        q_values, policy_state = self._network(network_obs, time_step.step_type, policy_state)
        logits = q_values
        distribution = shifted_categorical.ShiftedCategorical(logits=logits,
                                                              dtype=self._flat_action_spec.dtype,
                                                              shift=self._flat_action_spec.minimum)
        distribution = tf.nest.pack_sequence_as(self._action_spec, [distribution])
        return policy_step.PolicyStep(distribution, policy_state)