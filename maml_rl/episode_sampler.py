import gym
import torch
import numpy as np
import torch.nn.functional as F

from maml_rl.envs.subproc_vec_env import SubprocVecEnv
from maml_rl.episode import BatchEpisodes


class EpisodeSampler(object):
    def __init__(self, env_name, batch_size):
        self.env_name = env_name
        self.batch_size = batch_size
        self.env = gym.make(env_name)

    def sample(self, policy, params=None, gamma=0.95, device='cpu'):
        episodes = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)
        observation = self.env.reset()
        done = False
        while not done:
            with torch.no_grad():
                observation_tensor = torch.from_numpy(observation).to(device=device)
                try:
                    action_tmp = policy(observation_tensor.float(), params=params)
                    action_tmp.probs = F.relu(action_tmp.probs)
                    action_tensor = action_tmp.sample()
                    action = action_tensor.cpu().numpy()

                except RuntimeError:
                    action = np.array(self.env.action_space.sample())
                    pass

            new_observation, reward, done, _ = self.env.step(action)
            episode.append(observation, action, reward, 0)
            observations, batch_ids = new_observations, new_batch_ids
        return episodes
