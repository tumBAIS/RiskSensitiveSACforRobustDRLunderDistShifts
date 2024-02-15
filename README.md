# Risk-Sensitive Soft Actor-Critic for Robust Deep Reinforcement Learning under Distribution Shifts

This software uses a risk-sensitive version of Soft Actor-Critic for discrete actions to train and test a policy, represented by a neural network, that is robust against distribution shifts in an environment that represents typical contextual multi-stage stochastic combinatorial optimization problems. 

This method is proposed in:

> Tobias Enders, James Harrison, Maximilian Schiffer (2024). Risk-Sensitive Soft Actor-Critic for Robust Deep Reinforcement Learning under Distribution Shifts. arXiv preprint arXiv:2402.XXXXX.

All components (code, data, etc.) required to run the experiments reported in the paper are provided here. This includes the greedy and risk-neutral Soft Actor-Critic baseline algorithms as well as the robust benchmarks: manipulating the training data and entropy regularization.

## Overview
Data:
- The code to generate the data for the instances considered in the paper is available in the directory `data_generation`, alongside the code to manipulate the training data for the first benchmark.
- The generated data, which was used for the experiments reported in the paper, is available in the directory `data`.

Algorithms in the directory `algorithms`:
- The environment implementation is available in `environment.py`.
- The greedy baseline algorithm is implemented in `greedy.py` and can be executed using `main_greedy.py` with arguments as the exemplary ones provided in `args_greedy.txt` (see comments in `main_greedy.py` for explanations of the arguments).
- The risk-sensitive Soft Actor-Critic algorithm is implemented in the remaining code files and can be executed using `main.py` with arguments as the exemplary ones provided in `args_RL.txt` (see comments in `main.py` for explanations of the arguments). By adapting the arguments, the same code can be used to run risk-neutral Soft Actor-Critic and the benchmarks. The code in `trainer.py` and `sac_discrete.py` is partly based on code from this [GitHub repository](https://github.com/keiohta/tf2rl).

## Code Execution
To run the code, execute `python main.py @args_RL.txt` (analogously for the greedy algorithm). 

For the instances and hyperparameters reported in the paper, a GPU should be used. 
