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
