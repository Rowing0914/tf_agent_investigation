from __future__ import absolute_import, division, print_function
import collections, gin, tensorflow as tf

# from dqn_agent import q_policy

from tf_agents.agents import tf_agent
from tf_agents.policies import boltzmann_policy, epsilon_greedy_policy, greedy_policy
from tf_agents.policies import q_policy
from tf_agents.trajectories import trajectory
from tf_agents.utils import common, composite, eager_utils, nest_utils, value_ops

class DqnLossInfo(collections.namedtuple("DqnLossInfo", ("td_loss", "td_error"))):
    pass

def compute_td_targets(next_q_values, rewards, discounts):
    return tf.stop_gradient(rewards + discounts * next_q_values)

@gin.configurable
class DqnAgent(tf_agent.TFAgent):
    def __init__(self,
                 time_step_spec,
                 action_spec,
                 q_network,
                 optimizer,
                 observation_and_action_constraint_splitter=None,
                 epsilon_greedy=0.1,
                 n_step_update=1,
                 boltzmann_temperature=None,
                 emit_log_probability=False,
                 # Params for target network updates
                 target_q_network=None,
                 target_update_tau=1.0,
                 target_update_period=1,
                 td_errors_loss_fn=None,
                 gamma=1.0,
                 reward_scale_factor=1.0,
                 gradient_clipping=None,
                 # Params for debugging
                 debug_summaries=False,
                 summarize_grads_and_vars=False,
                 train_step_counter=None,
                 name=None):
        tf.Module.__init__(self, name=name)
        self._check_action_spec(action_spec)

        if epsilon_greedy is not None and boltzmann_temperature is not None:
            raise ValueError(
                'Configured both epsilon_greedy value {} and temperature {}, '
                'however only one of them can be used for exploration.'.format(
                    epsilon_greedy, boltzmann_temperature))

        self._observation_anc_action_constraint_splitter = (
            observation_and_action_constraint_splitter
        )
        self._q_network = q_network
        q_network.create_variables()
        if target_q_network:
            target_q_network.create_variables()
        self._target_q_network = common.maybe_copy_target_network_with_checks(
            self._q_network, target_q_network, "TargetQNetwork"
        )

        self._epsilon_greedy = epsilon_greedy
        self._n_step_update = n_step_update
        self._boltzmann_temperature = boltzmann_temperature
        self._optimizer = optimizer
        self._td_error_loss_fn = (
                td_errors_loss_fn or common.element_wise_huber_loss
        )
        self._gamma = gamma
        self._reward_scale_factor = reward_scale_factor
        self._gradient_clipping = gradient_clipping
        self._update_target = self._get_target_updater(
            target_update_tau, target_update_period
        )

        policy, collect_policy = self._setup_policy(time_step_spec,
                                                    action_spec,
                                                    boltzmann_temperature,
                                                    emit_log_probability)

        if q_network.state_spec and n_step_update != 1:
            raise NotImplementedError(
                'DqnAgent does not currently support n-step updates with stateful '
                'networks (i.e., RNNs), but n_step_update = {}'.format(n_step_update))

        train_sequence_length = (
            n_step_update + 1 if not q_network.state_spec else None
        )

        super(DqnAgent, self).__init__(
            time_step_spec,
            action_spec,
            policy,
            collect_policy,
            train_sequence_length=train_sequence_length,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=train_step_counter)

    def _check_action_spec(self, action_spec):
        flat_action_spec = tf.nest.flatten(action_spec)
        self._num_actions = [
            spec.maximum - spec.minimum + 1 for spec in flat_action_spec
        ]

        if len(flat_action_spec) > 1 or flat_action_spec[0].shape.rank > 1:
            raise ValueError("only one dim actions are supported now")

        if not all(spec.minimum == 0 for spec in flat_action_spec):
            raise ValueError("action specs should have minimum of 0, but saw: {}".format(
                [spec.minimum for spec in flat_action_spec]))

    def _setup_policy(self, time_step_spec, action_spec, boltzmann_temperature, emit_log_probability):
        policy = q_policy.QPolicy(
            time_step_spec,
            action_spec,
            q_network=self._q_network,
            emit_log_probability=emit_log_probability,
            observation_and_action_constraint_splitter=(self._observation_anc_action_constraint_splitter))
        if boltzmann_temperature is not None:
            collect_policy = boltzmann_policy.BoltzmannPolicy(
                policy, temperature=self._boltzmann_temperature)
        else:
            collect_policy = epsilon_greedy_policy.EpsilonGreedyPolicy(
                policy, epsilon=self._epsilon_greedy)
        policy = greedy_policy.GreedyPolicy(policy)

        target_policy = q_policy.QPolicy(
            time_step_spec,
            action_spec,
            q_network=self._target_q_network,
            observation_and_action_constraint_splitter=(self._observation_anc_action_constraint_splitter))
        self._target_greedy_policy = greedy_policy.GreedyPolicy(target_policy)
        return policy, collect_policy

    def _initialize(self):
        common.soft_variables_update(self._q_network.variables, self._target_q_network.variables, tau=1.0)

    def _get_target_updater(self, tau=1.0, period=1):
        with tf.name_scope("update_targets"):
            def update():
                return common.soft_variables_update(
                    self._q_network.variables, self._target_q_network.variables, tau)
        return common.Periodically(update, period, "periodic_update_targets")

    def _experience_to_transitions(self, experience):
        transitions = trajectory.to_transition(experience)
        if not self._q_network.state_spec:
            transitions = tf.nest.map_structure(lambda x: composite.squeeze(x, 1), transitions)

        time_steps, policy_steps, next_time_steps = transitions
        actions = policy_steps.action
        return time_steps, actions, next_time_steps

    def _train(self, experience, weights):
        with tf.GradientTape() as tape:
            loss_info = self._loss(experience,
                                   td_errors_loss_fn=self._td_error_loss_fn,
                                   gamma=self._gamma,
                                   reward_scale_factor=self._reward_scale_factor,
                                   weights=weights)
        tf.debugging.check_numerics(loss_info[0], "Loss is inf or Nan")
        variables_to_train = self._q_network.trainable_variables
        assert list(variables_to_train), "No variables in the agent's q_network."
        grads = tape.gradient(loss_info.loss, variables_to_train)
        grads_and_vars = tuple(zip(grads, variables_to_train))
        if self._gradient_clipping is not None:
            grads_and_vars = eager_utils.clip_gradient_norms(grads_and_vars, self._gradient_clipping)
        if self._summarize_grads_and_vars:
            eager_utils.add_variables_summaries(grads_and_vars, self.train_step_counter)
            eager_utils.add_gradients_summaries(grads_and_vars, self.train_step_counter)

        self._optimizer.apply_gradients(grads_and_vars, global_step=self.train_step_counter)
        self._update_target()
        return loss_info

    def _loss(self,
              experience,
              td_errors_loss_fn=common.element_wise_huber_loss,
              gamma=1.0,
              reward_scale_factor=1.0,
              weights=None):
        self._check_trajectory_dimensions(experience)

        if self._n_step_update == 1:
            time_steps, actions, next_time_steps = self._experience_to_transitions(experience)
        else:
            first_two_steps = tf.nest.map_structure(lambda x: x[:, :2], experience)
            last_two_steps = tf.nest.map_structure(lambda x: x[:, -2:], experience)
            time_steps, actions, _ = self._experience_to_transitions(first_two_steps)
            _, _, next_time_steps = self._experience_to_transitions(last_two_steps)

        with tf.name_scope("loss"):
            q_values = self._compute_q_values(time_steps, actions)
            next_q_values = self._compute_next_q_values(next_time_steps)

            if self._n_step_update == 1:
                td_targets = compute_td_targets(next_q_values,
                                               rewards=reward_scale_factor*next_time_steps.reward,
                                               discounts=gamma*next_time_steps.discount)
            else:
                rewards = reward_scale_factor * experience.reward[:, :-1]
                discounts = gamma * experience.discount[:, :-1]
                td_targets = value_ops.discounted_return(rewards=rewards,
                                                         discounts=discounts,
                                                         final_value=next_q_values,
                                                         time_major=False,
                                                         provide_all_returns=False)
            valid_mask = tf.cast(~time_steps.is_last(), tf.float32)
            td_error = valid_mask * (td_targets - q_values)
            td_loss = valid_mask * td_errors_loss_fn(td_targets, q_values)

            if nest_utils.is_batched_nested_tensors(time_steps, self.time_step_spec, num_outer_dims=2):
                td_loss = tf.reduce_sum(input_tensor=td_loss, axis=1)

            if weights is not None:
                td_loss *= weights

            loss = tf.reduce_mean(input_tensor=td_loss)

            if self._q_network.losses:
                loss = loss + tf.reduce_mean(self._q_network.losses)

            with tf.name_scope("Losses/"):
                tf.compat.v2.summary.scalar(name="loss", data=loss, step=self.train_step_counter)

            if self._summarize_grads_and_vars:
                with tf.name_scope("Variables/"):
                    for var in self._q_network.trainable_weights:
                        tf.compat.v2.summary.historgram(name=var.name.replace(":", "_"),
                                                        data=var,
                                                        step=self.train_step_counter)

            if self._debug_summaries:
                diff_q_values = q_values - next_q_values
                common.generate_tensor_summaries("td_error", td_error, self.train_step_counter)
                common.generate_tensor_summaries("td_loss", td_loss, self.train_step_counter)
                common.generate_tensor_summaries("q_values", q_values, self.train_step_counter)
                common.generate_tensor_summaries("next_q_values", next_q_values, self.train_step_counter)
                common.generate_tensor_summaries("diff_q_values", diff_q_values, self.train_step_counter)
            return tf_agent.LossInfo(loss, DqnLossInfo(td_loss=td_loss, td_error=td_error))

    def _compute_q_values(self, time_steps, actions):
        network_observation = time_steps.observation

        if self._observation_anc_action_constraint_splitter:
            network_observation, _ = self._observation_anc_action_constraint_splitter(network_observation)

        q_values, _ = self._q_network(network_observation, time_steps.step_type)
        multi_dim_actions = self._action_spec.shape.rank > 0
        return common.index_with_actions(q_values,
                                         tf.cast(actions, dtype=tf.int32),
                                         multi_dim_actions=multi_dim_actions)

    def _compute_next_q_values(self, next_time_steps):
        network_observation = next_time_steps.observation

        if self._observation_anc_action_constraint_splitter:
            network_observation, _ = self._observation_anc_action_constraint_splitter(network_observation)

        next_target_q_values, _ = self._target_q_network(network_observation, next_time_steps.step_type)
        batch_size = (next_target_q_values.shape[0] or tf.shape(next_target_q_values)[0])
        dummy_state = self._target_greedy_policy.get_initial_state(batch_size)
        greedy_actions = self._target_greedy_policy.action(next_time_steps, dummy_state).action
        multi_dim_actions = tf.nest.flatten(self._action_spec)[0].shape.rank > 0
        return common.index_with_actions(next_target_q_values,
                                         greedy_actions,
                                         multi_dim_actions=multi_dim_actions)