from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin, tensorflow as tf
from tf_agents.agents import tf_agent
from tf_agents.policies import epsilon_greedy_policy
from tf_agents.trajectories import trajectory
from tf_agents.utils import composite, common

from sample.policy import Policy

@gin.configurable
class Agent(tf_agent.TFAgent):
    def __init__(self,
                 time_step_spec,
                 action_spec,
                 network,
                 optimizer,
                 epsilon_greedy=0.1,
                 n_step_update=1,
                 name=None):
        tf.Module.__init__(self, name=name)
        self._network = network
        self._optimizer = optimizer
        self._epsilon_greedy = epsilon_greedy
        policy, collect_policy = self._setup_policy(time_step_spec, action_spec)
        self._collect_policy = collect_policy

        train_sequence_length = (n_step_update + 1 if not network.state_spec else None)

        super(Agent, self).__init__(
            time_step_spec,
            action_spec,
            policy,
            collect_policy,
            train_sequence_length=train_sequence_length
        )

    def _setup_policy(self, time_step_spec, action_spec):
        policy = Policy(time_step_spec,
                        action_spec,
                        network=self._network)
        collect_policy = epsilon_greedy_policy.EpsilonGreedyPolicy(policy, epsilon=self._epsilon_greedy)
        return policy, collect_policy

    def _initialize(self):
        pass

    def _train(self, experience, weights):
        with tf.GradientTape() as tape:
            loss_info = self._loss(experience)
        grads = tape.gradient(loss_info.loss, self._network.trainable_weights)
        grads_and_vars = tuple(zip(grads, self._network.trainable_weights))
        self._optimizer.apply_gradients(grads_and_vars, global_step=self.train_step_counter)
        return loss_info

    def _experience_to_transitions(self, experience):
        transitions = trajectory.to_transition(experience)
        if not self._network.state_spec:
            transitions = tf.nest.map_structure(lambda x: composite.squeeze(x, 1), transitions)

        time_steps, policy_steps, next_time_steps = transitions
        actions = policy_steps.action
        return time_steps, actions, next_time_steps

    def _loss(self, experience):
        time_steps, actions, next_time_steps = self._experience_to_transitions(experience)
        print(time_steps, actions, next_time_steps)
        with tf.name_scope("loss"):
            predictions = self._compute_q_values(time_steps, actions)
            loss = tf.reduce_mean(tf.cast(actions, tf.float32) - predictions)

        return tf_agent.LossInfo(loss=loss, extra=tf.constant(0))

    def _compute_q_values(self, time_steps, actions):
        network_obs = time_steps.observation
        q_values, _ = self._network(network_obs, time_steps.step_type)
        return common.index_with_actions(
            q_values,
            tf.cast(actions, dtype=tf.int32))