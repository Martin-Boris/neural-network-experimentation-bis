import imageio
import torch
from stable_baselines3 import PPO

#### NN CLASS
import numpy as np

from stable_baselines.mad_pod_racing import MapPodRacing

import torch
import torch.nn as nn
from torch.distributions import Normal


class DiagGaussianDistribution:
    def __init__(self, mean, log_std_):
        self.mean = mean
        self.log_std = log_std_
        self.std = torch.exp(log_std_)
        self.dist = Normal(mean, self.std)

    def sample(self):
        return self.dist.rsample()  # For reparameterization

    def mode(self):
        return self.mean

    def log_prob(self, actions_):
        return self.dist.log_prob(actions_).sum(axis=-1)

    def entropy(self):
        return self.dist.entropy().sum(axis=-1)


class PPOPolicyVanilla(nn.Module):
    def __init__(self, input_dim=9, output_dim=2, low=torch.tensor([0.0, 0.0], dtype=torch.float32),
                 high=torch.tensor([2 * np.pi, 1.0], dtype=torch.float32)):
        super().__init__()
        self.low = low
        self.high = high
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)

        # Actor head: outputs mean of Gaussian
        self.mean_head = nn.Linear(128, output_dim)

        # Learnable log_std (same shape as action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mean = self.mean_head(x)
        std = torch.exp(self.log_std)

        value = self.value_head(x)
        return mean, std, value

    def rescale_action(self, action):
        # From Gaussian distribution to [low, high]
        return self.low + 0.5 * (action + 1.0) * (self.high - self.low)

    def predict(self, obs_np: np.ndarray, log_std_) -> np.ndarray:
        obs_tensor = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            mean = self.forward(obs_tensor)  # .squeeze(0)
            log_std_expanded = log_std_.expand_as(mean)

            '''# Build the Diagonal Gaussian distribution
            dist = DiagGaussianDistribution(mean, log_std_expanded)
            action = dist.mode()'''
            action = mean

            # Squash
            action = torch.tanh(action)

        # Rescale
        scaled_action = self.rescale_action(action.squeeze(0))

        return scaled_action.numpy()

    def get_action(self, obs, deterministic=True):
        x = torch.tanh(self.fc1(obs))
        x = torch.tanh(self.fc2(x))

        mean = self.mean_head(x)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)

        if deterministic:
            action = mean
        else:
            action = dist.rsample()

        # Clip directly to [low, high]
        action = torch.clamp(action, self.low, self.high)

        value_ = self.value_head(x)

        return action, value_

    def feed_nn_parameter(self, sb3_policy):
        # Shared layers
        self.fc1.weight.data = sb3_policy.mlp_extractor.policy_net[0].weight.data.clone()
        self.fc1.bias.data = sb3_policy.mlp_extractor.policy_net[0].bias.data.clone()

        self.fc2.weight.data = sb3_policy.mlp_extractor.policy_net[2].weight.data.clone()
        self.fc2.bias.data = sb3_policy.mlp_extractor.policy_net[2].bias.data.clone()

        # Actor mean head
        self.mean_head.weight.data = sb3_policy.action_net.weight.data.clone()
        self.mean_head.bias.data = sb3_policy.action_net.bias.data.clone()

        # Log std
        self.log_std.data = sb3_policy.log_std.data.clone()

        # Critic head
        self.value_head.weight.data = sb3_policy.value_net.weight.data.clone()
        self.value_head.bias.data = sb3_policy.value_net.bias.data.clone()


### MODEL WEIGHT LOAD

sb3_model = PPO.load("./runs/PPO_22/model")
print(sb3_model.policy)

obs_dim = sb3_model.policy.observation_space.shape[0]
action_dim = sb3_model.policy.action_space.shape[0]

nn_model = PPOPolicyVanilla(obs_dim, action_dim)
nn_model.feed_nn_parameter(sb3_model.policy)
nn_model.eval()

env = MapPodRacing()

score = 0
done = False
truncated = False
observation, info = env.reset(seed=1)
frames = [env.render()]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

while not done:
    # actions, std, value = nn_model.forward(torch.from_numpy(observation).to(device=device))
    actions, value = nn_model.get_action(torch.from_numpy(observation).to(device=device))
    print(actions)
    observation_, reward, done, truncated, info = env.step(actions.detach().numpy())
    score += reward
    frames.append(env.render())
    observation = observation_
print("score ", str(score))
imageio.mimsave("mad_pod_episode.gif", frames, fps=10)
