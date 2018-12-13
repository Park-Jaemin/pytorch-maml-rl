# Reinforcement Learning with Model-Agnostic Meta-Learning (MAML)
[Model-Agnostic Meta-Learning (MAML)](https://arxiv.org/abs/1703.03400) pytorch implementation for Atari environment.

## Getting started
To avoid any conflict with your existing Python setup, and to keep this project self-contained, it is suggested to work in a virtual environment with [`virtualenv`](http://docs.python-guide.org/en/latest/dev/virtualenvs/). To install `virtualenv`:
```
pip install --upgrade virtualenv
```
Create a virtual environment, activate it and install the requirements in [`requirements.txt`](requirements.txt).
```
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## Run meta-learning
You can use the [`main.py`](main.py) script in order to run reinforcement learning experiments with MAML. This script was tested with Python 3.6.  

To run MAML
```
python3 main.py --env-name BankHeist-ram-v0
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
 tensorboard --logdir='./logs'
 ```
 
 ## Run reinforcement
To run REINFORCEMENT for an environment
```
python3 reinforcement.py --env-name Alien-ram-v0 --trained-model True --trained-model-dir saves/Atari/policy-200.pt
```
* If you check True for trained model, RL starts with the model in trained_model_dir
 If you check False, RL starts with random initialized model

To see graph of the log
```
tensorboard --logdir='./reinforcement/logs/Atari/'
```

## Algorithm
TRPO is used for meta-learner.  
REINFORCEMENT is used for learner.  

## Result
Random Initialize  
![image](https://user-images.githubusercontent.com/19935323/49953319-357ad680-ff41-11e8-845a-7d153885f896.png)

MAML  
![image](https://user-images.githubusercontent.com/19935323/49953295-28f67e00-ff41-11e8-9380-2ac84594fb4a.png)

## References
This project is forked from [tristandeleu/pytorch-maml-rl](https://github.com/tristandeleu/pytorch-maml-rl).  
Most codes are from this repository.  
