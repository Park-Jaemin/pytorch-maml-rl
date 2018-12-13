# Reinforcement Learning with Model-Agnostic Meta-Learning (MAML)

This repository contains code for MAML on Atari environment.

## Getting started
To avoid any conflict with your existing Python setup, and to keep this project self-contained, it is suggested to work in a virtual environment with [`virtualenv`](http://docs.python-guide.org/en/latest/dev/virtualenvs/). To install `virtualenv`:
```
pip install --upgrade virtualenv
```
Create a virtual environment, activate it and install the requirements in [`requirements.txt`](requirements.txt).
```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
You can use the [`main.py`](main.py) script in order to run reinforcement learning experiments with MAML. This script was tested with Python 3.6.  

To run MAML
```
python main.py --env-name BankHeist-ram-v0
```
* This code trains model for Atari games with 128 states, 18 actions.  
 You can change environments by changing maml_rl/sampler.py  
* *env_name* is just a placeholder who has same states, same actions with other envs.  
 default *env_name* is 'BankHeist-ram-v0'  
* There are some issues for running on gpu.  
 For now, you can run only on cpu.
* Learned model is saved on saves/*env_name* as .pt file.  
* Training log is saved on logs/*env_name*  
 To see graph of the log
 ```
 tensorboard --logdir='./logs' --port=6006
 ```
To run REINFORCEMENT for an environment


## Algorithm
TRPO is used for meta-learner.  
REINFORCEMENT is used for learner.  

## References
This project is forked from [tristandeleu/pytorch-maml-rl](https://github.com/tristandeleu/pytorch-maml-rl).  
Most codes are from this repository.  
