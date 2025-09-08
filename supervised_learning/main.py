import os
import random
import time
from dataclasses import dataclass
import gymnasium as gym
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
import wandb

from supervised_learning.mad_pod_racing import MapPodRacing, supervised_action_choose_from_angle, \
    supervised_action_choose


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None

    # Algorithm specific arguments
    env_id: str = "gymnasium_env/MapPodRacing-v0"  # "CartPole-v1"
    """the id of the environment"""
    total_games: int = 3000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5
    """the learning rate of the optimizer"""
    hidden_size: int = 128
    """the size of the hidden layer"""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, input_dim = 3,output_dim=1,chkpt_dir='tmp/'):
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(input_dim).prod(), 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, output_dim), std=0.01),
        )


    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action = self.actor(x)
        return action

    def save_checkpoint(self, filename):
        checkpoint_file = os.path.join('tmp/', filename)
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self,filename):
        checkpoint_file = os.path.join('tmp/', filename)
        self.load_state_dict(torch.load(checkpoint_file,weights_only=True))

if __name__ == "__main__":
    gym.register(
        id="gymnasium_env/MapPodRacing-v0",
        entry_point=MapPodRacing
    )
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env = gym.make(args.env_id)
    agent = Agent(input_dim=env.observation_space.shape[0], output_dim=env.action_space.shape[0]).to(device)
    #agent.load_checkpoint('experimentation_model.pth')
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()

    cp_completions = []

    for i in range(args.total_games):
        observation, _ = env.reset()
        done = False
        truncated = False
        score = 0
        completion_cp = 0
        n_steps = 0
        cp_completion = 0
        while not done and not truncated:
            action = agent.get_action_and_value(torch.from_numpy(observation).to(device=device))
            observation_, reward, done, truncated, info = env.step(action.detach())
            n_steps += 1
            cp_completion = info["cp_completion"]
            score += reward
            observation = observation_
            loss = loss_fn(action,torch.tensor([supervised_action_choose(observation)],dtype=torch.float32))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        cp_completions.append(cp_completion)
        print("episode : ",i," score : ",score, " round : ", n_steps, " cp_completion : ", cp_completion)
        writer.add_scalar("charts/episodic_return", score, i)
        writer.add_scalar("charts/episodic_length", n_steps, i)
        writer.add_scalar("charts/cp_completion", cp_completion, i)

    agent.save_checkpoint('model_first_learning.pth')
    score = 0
    done = False
    truncated = False
    observation, info = env.reset()
    print(observation)
    frames = [env.render()]

    while not done and not truncated:
        action = agent.get_action_and_value(torch.from_numpy(observation).to(device=device))

        observation_, reward, done, truncated, info = env.step(action.detach())
        score += reward
        frames.append(env.render())
        observation = observation_
    print("score ", str(score))
    imageio.mimsave("mad_pod_episode.gif", frames, fps=10)