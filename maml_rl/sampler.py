import gym
import torch
import multiprocessing as mp
import numpy as np
import torch.nn as nn

from maml_rl.envs.subproc_vec_env import SubprocVecEnv
from maml_rl.episode import BatchEpisodes


def make_env(env_name):
    def _make_env():
        return gym.make(env_name)
    return _make_env


class BatchSampler(object):
    def __init__(self, env_name, batch_size, num_workers=mp.cpu_count() - 1):
        self.env_name = env_name
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.queue = mp.Queue()
        self.envs = SubprocVecEnv([make_env(env_name) for _ in range(num_workers)],
            queue=self.queue)
        self._env = gym.make(env_name)

    def sample(self, policy, params=None, gamma=0.95, device='cpu'):
        episodes = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)
        for i in range(self.batch_size):
            self.queue.put(i)
        for _ in range(self.num_workers):
            self.queue.put(None)
        observations, batch_ids = self.envs.reset()
        dones = [False]
        while (not all(dones)) or (not self.queue.empty()):
            with torch.no_grad():
                observations_tensor = torch.from_numpy(observations).to(device=device)
                try:
                    action_tmp = policy(observations_tensor.float(), params=params)
                    action_tmp.probs = nn.functional.relu(action_tmp.probs)
                    actions_tensor = action_tmp.sample()
                    actions = actions_tensor.cpu().numpy()

                except RuntimeError:
                    actions = np.array([self.envs.action_space.sample() for _ in range(self.num_workers)])
                    pass

            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
            episodes.append(observations, actions, rewards, batch_ids)
            observations, batch_ids = new_observations, new_batch_ids
        return episodes

    def reset_task(self, task):
        tasks = [task for _ in range(self.num_workers)]
        reset = self.envs.reset_task(tasks)
        return all(reset)

    def sample_tasks(self, num_tasks):
        # Atari games that has 128 observations, 18 actions
        tasks = ['BankHeist-ram-v0', 'Berzerk-ram-v0', 'IceHockey-ram-v0', 'Jamesbond-ram-v0', 'Kangaroo-ram-v0', 'Krull-ram-v0', 'Riverraid-ram-v0', 'RoadRunner-ram-v0', 'Seaquest-ram-v0', 'Tennis-ram-v0', 'Venture-ram-v0']
        return np.random.choice(tasks, num_tasks, replace=False)