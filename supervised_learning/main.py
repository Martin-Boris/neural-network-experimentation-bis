import os
import random
import time
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import wandb

from supervised_learning.mad_pod_racing import MapPodRacing


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

    # Algorithm specific arguments
    env_id: str = "gymnasium_env/MapPodRacing-v0"  # "CartPole-v1"
    """the id of the environment"""
    total_games: int = 1000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-3
    """the learning rate of the optimizer"""
    num_steps: int = 512
    """the number of steps to run in each environment per policy rollout"""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(env.action_space.shape).prod(), 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, env.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action


if __name__ == "__main__":
    gym.register(
        id="gymnasium_env/MapPodRacing-v0",
        entry_point=MapPodRacing
    )
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        sync_tensorboard=True,
        config=vars(args),
        name=run_name,
        monitor_gym=True,
        save_code=True,
    )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env = gym.make(args.env_id)
    agent = Agent(env).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    n_steps = 0
    for i in range(args.total_games):
        observation, _ = env.reset()
        done = False
        truncated = False
        score = 0
        completion_cp = 0
        while not done and not truncated:
            action, prob, val = agent.get_action_and_value(observation)
            observation_, reward, done, truncated, info = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            observation = observation_
