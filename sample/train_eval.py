from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import app
from absl import flags
from absl import logging

import gin
import tensorflow as tf

from sample.agent import Agent
from sample.network import Network

from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import py_metrics
from tf_agents.metrics import tf_metrics
from tf_agents.policies import py_tf_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

flags.DEFINE_string('root_dir', "./",
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer('num_iterations', 100000,
                     'Total number train/eval iterations to perform.')
flags.DEFINE_bool('use_ddqn', False,
                  'If True uses the DdqnAgent instead of the DqnAgent.')
FLAGS = flags.FLAGS

@gin.configurable
def train_eval(root_dir,
               env_name="CartPole-v0",
               agent_class=Agent,
               num_iterations=10000,
               initial_collect_steps=1000,
               collect_steps_per_iteration=1,
               epsilon_greedy=0.1,
               replay_buffer_capacity=10000,
               train_steps_per_iteration=1,
               batch_size=32):
    global_step = tf.compat.v1.train.get_or_create_global_step()
    tf_env = tf_py_environment.TFPyEnvironment(suite_gym.load(env_name))
    eval_py_env = suite_gym.load(env_name)
    network = Network(input_tensor_spec=tf_env.time_step_spec().observation,
                      action_spec=tf_env.action_spec())
    tf_agent = agent_class(
        time_step_spec=tf_env.time_step_spec(),
        action_spec=tf_env.action_spec(),
        network=network,
        optimizer=tf.compat.v1.train.AdamOptimizer(),
        epsilon_greedy=epsilon_greedy)
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        tf_agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=replay_buffer_capacity)
    eval_py_policy = py_tf_policy.PyTFPolicy(tf_agent.policy)
    replay_observer = [replay_buffer.add_batch]
    initial_collect_policy = random_tf_policy.RandomTFPolicy(tf_env.time_step_spec(),
                                                             tf_env.action_spec())
    initial_collect_op = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        initial_collect_policy,
        observers=replay_observer,
        num_steps=initial_collect_steps).run()
    collect_policy = tf_agent.collect_policy
    collect_op = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        collect_policy,
        observers=replay_observer,
        num_steps=collect_steps_per_iteration
    ).run()
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2
    ).prefetch(3)

    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
    experience, _ = iterator.get_next()
    train_op = common.function(tf_agent.train)(experience=experience)

    init_agent_op = tf_agent.initialize()

    with tf.compat.v1.Session() as sess:
        sess.run(iterator.initializer)
        common.initialize_uninitialized_variables(sess)

        sess.run(init_agent_op)
        sess.run(initial_collect_op)

        global_step_val = sess.run(global_step)

        collect_call = sess.make_callable(collect_op)
        global_step_call = sess.make_callable(global_step)
        train_step_call = sess.make_callable(train_op)

        for _ in range(num_iterations):
            collect_call()
            for _ in range(train_steps_per_iteration):
                loss_info_value, _ = train_step_call()

            global_step_val = global_step_call()
            logging.info("step = %d, loss = %d", global_step_val, loss_info_value)


def main(_):
    logging.set_verbosity(logging.INFO)
    tf.compat.v1.enable_resource_variables()
    agent_class = Agent
    train_eval(root_dir=FLAGS.root_dir,
               agent_class=agent_class,
               num_iterations=FLAGS.num_iterations)

if __name__ == '__main__':
    # flags.mark_flag_as_required("root_dir")
    app.run(main=main)