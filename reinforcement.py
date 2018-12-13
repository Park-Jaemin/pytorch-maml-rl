import gym
import numpy as np
import torch

from maml_rl.policies import CategoricalMLPPolicy
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.sampler import BatchSampler
from tensorboardX import SummaryWriter


def total_rewards(episodes_rewards, aggregation=torch.mean):
    rewards = torch.mean(torch.stack([aggregation(torch.sum(rewards, dim=0))
        for rewards in episodes_rewards], dim=0))
    return rewards.item()


def main(args):
    writer = SummaryWriter('./reinforcement/logs{0}'.format(args.output_folder))
    save_folder = './reinforcement/saves/{0}'.format(args.output_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    sampler = BatchSampler(env_name=args.env_name, batch_size=args.batch_size, num_workers=args.num_workers)

    policy = CategoricalMLPPolicy(
        int(np.prod(sampler.env.observation_space.shape)),
        sampler.envs.action_space.n,
        hidden_sizes=(args.hidden_size,) * args.num_layers)
    baseline = LinearFeatureBaseline(
        int(np.prod(sampler.env.observation_space.shape)))

    if args.trained_model:
        check_point = torch.load(args.trained_model_dir)
        policy.load_state_dict(check_point)

    iter = 0
    while True:
        iter = iter + 1
        print(iter, 'th iter is running')

        sampler.reset_task(args.env_name)
        episodes = sampler.sample(params=policy.staet_dict())

        # Tensorboard
        writer.add_scalar('total_rewards',
            total_rewards([ep.rewards for ep in episodes]), iter)

        # Save policy network
        with open(os.path.join(save_folder, 'policy-{0}.pt'.format(iter)), 'wb') as f:
            torch.save(policy.state_dict(), f)


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML)')

    # General
    parser.add_argument('--env-name', type=str, default='BankHeist-ram-v0',
        help='name of the environment')
    parser.add_argument('--gamma', type=float, default=0.95,
        help='value of the discount factor gamma')
    parser.add_argument('--tau', type=float, default=1.0,
        help='value of the discount factor for GAE')
    parser.add_argument('--batch-size', type=int, default=20,
        help='batch size')
    parser.add_argument('--lr', type=float, default=0.1,
        help='learning rate gradient update')

    # Policy network (relu activation function)
    parser.add_argument('--trained-model', type=bool, default=False,
        help='If true, load trained model. If false, random initialize')
    parser.add_argument('--trained-model-dir', type=str, default='saves/Atari/policy-100.pt',
        help='directory of trained model')
    parser.add_argument('--hidden-size', type=int, default=100,
        help='number of hidden units per layer')
    parser.add_argument('--num-layers', type=int, default=2,
        help='number of hidden layers')

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='Atari',
        help='name of the output folder')
    parser.add_argument('--device', type=str, default='cpu',
        help='set the device (cpu or cuda)')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling')

    args = parser.parse_args()

    # Create logs and saves folder if they don't exist
    if not os.path.exists('./reinforcement/logs'):
        os.makedirs('./reinforcement/logs')
    if not os.path.exists('./reinforcement/saves'):
        os.makedirs('./reinforcement/saves')


    main(args)
