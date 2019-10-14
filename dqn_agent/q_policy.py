from __future__ import absolute_import, division, print_function

import gin, numpy as np, tensorflow as tf

from tf_agents.distributions import shifted_categorical
from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step

@gin.configurable
class QPolicy(tf_policy.Base):
    def __init__(self,
                 time_step_spec,
                 action_spec,
                 q_network,
                 observation_and_action_constraint_splitter=None,
                 emit_log_probability=None,
                 name=None):
        self._observation_and_action_constraint_splitter = (observation_and_action_constraint_splitter)
        network_action_spec = getattr(q_network, "action_spec", None)

        if network_action_spec is not None:
            if not action_spec.is_compatible_with(network_action_spec):
                raise ValueError("action_spec must be compatible with q_network.action_spec"
                                 "instead, we got action_spec=%s, q_network.action_spec=%s"%(action_spec,
                                                                                             network_action_spec))

        flat_action_spec = tf.nest.flatten(action_spec)
        if len(flat_action_spec) > 1:
            raise NotImplementedError("action_spec can only contain a single BoundedTensorSpec")

        self._flat_action_spec = flat_action_spec[0]
        q_network.create_variables()
        self._q_network = q_network
        super(QPolicy, self).__init__(time_step_spec,
                                      action_spec,
                                      policy_state_spec=q_network.state_spec,
                                      clip=False,
                                      emit_log_probability=emit_log_probability,
                                      name=name)

    @property
    def observation_and_action_contraint_splitter(self):
        return self._observation_and_action_constraint_splitter

    def _variables(self):
        return self._q_network.variables

    def _distribution(self, time_step, policy_state):
        network_observation = time_step.observation

        if self._observation_and_action_constraint_splitter:
            network_observation, mask = (self._observation_and_action_constraint_splitter(network_observation))

        q_values, policy_state = self._q_network(network_observation, time_step.step_type, policy_state)

        if self._flat_action_spec.shape.rank == 1:
            q_values = tf.expand_dims(q_values, -2)

        logits = q_values

        if self._observation_and_action_constraint_splitter:
            if self._flat_action_spec.shape.rank == 1:
                mask = tf.expand_dims(mask, -2)

            neg_inf = tf.constat(-np.inf, dtype=logits.dtype)
            logits = tf.compat.v2.where(tf.cast(mask, tf.bool), logits, neg_inf)

        distribution = shifted_categorical.ShiftedCategorical(logits=logits,
                                                              dtype=self._flat_action_spec.dtype,
                                                              shift=self._flat_action_spec.minimum)

        distribution = tf.nest.pack_sequence_as(self._action_spec, [distribution])
        return policy_step.PolicyStep(distribution, policy_state)
