# tf_agent_investigation
this is a dummy codebase which is supposed to play with tf-agent as well as understand how it works.
So in `dqn_agent` dir, you'll see the exact same code as the original tf-agent implementation.
It is just because I wanted to replicate it myself.

On the other hand, in `sample`, I have briefly scratched a sample code including
- training script: `train_eval.py`
- policy: `policy.py` => inherits `tf_policy.Base`
- network: `network.py` => inherits `tf_agent.network.Network`
- agent: `agent.py` => inherits `tf_agent.TFAgent`

with above-listed scripts, i think we can create our own RL algo based on the HUGE support of utils provided by `tf-agent` ofcourse.

## Note
- if you want to change the behaviour of agent, in other words, if you want to change the logic to select an action, then modify(override from your subclass) `tf_policy.Base._action` API. this uses `_distribution` API, which is supposed to be implemented by your subclass.
- `tf_agent.TFAgent` mainly is responsible for training the model(the neural network/other models used in `network.Network`) by `train` API.
- `replay_buffer` has to know the placeholders of collected data(trajectories), which is supposed to be provided by `collect_policy` in `tf_agent.TFAgent`.
