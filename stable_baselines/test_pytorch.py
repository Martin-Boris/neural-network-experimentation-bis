import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
import imageio

from stable_baselines.mad_pod_racing import MapPodRacing


class VanillaContinuousPolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        # Shared layers
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)

        # Actor head: outputs mean of Gaussian
        self.mean_head = nn.Linear(128, action_dim)

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


# Copy weights
def copy_weights_continuous(sb3_policy, vanilla_model):
    # Shared layers
    vanilla_model.fc1.weight.data = sb3_policy.mlp_extractor.policy_net[0].weight.data.clone()
    vanilla_model.fc1.bias.data = sb3_policy.mlp_extractor.policy_net[0].bias.data.clone()

    vanilla_model.fc2.weight.data = sb3_policy.mlp_extractor.policy_net[2].weight.data.clone()
    vanilla_model.fc2.bias.data = sb3_policy.mlp_extractor.policy_net[2].bias.data.clone()

    # Actor mean head
    vanilla_model.mean_head.weight.data = sb3_policy.action_net.weight.data.clone()
    vanilla_model.mean_head.bias.data = sb3_policy.action_net.bias.data.clone()

    # Log std
    vanilla_model.log_std.data = sb3_policy.log_std.data.clone()

    # Critic head
    vanilla_model.value_head.weight.data = sb3_policy.value_net.weight.data.clone()
    vanilla_model.value_head.bias.data = sb3_policy.value_net.bias.data.clone()


# Load the model
# model = PPO.load("./runs/PPO_6/model")
model = PPO.load("./runs/PPO_17/model")

# Get SB3 policy
sb3_policy = model.policy

# Observation and action dimensions
obs_dim = sb3_policy.observation_space.shape[0]
action_dim = sb3_policy.action_space.shape[0]

# Instantiate vanilla model
model = VanillaContinuousPolicyNetwork(obs_dim, action_dim)

copy_weights_continuous(sb3_policy, model)

env = MapPodRacing()

score = 0
done = False
truncated = False
observation, info = env.reset()
frames = [env.render()]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

while not done:
    mean, std, value = model.forward(torch.from_numpy(observation).to(device=device))
    observation_, reward, done, truncated, info = env.step(mean.detach().numpy())
    score += reward
    frames.append(env.render())
    observation = observation_
print("score ", str(score))
imageio.mimsave("mad_pod_episode.gif", frames, fps=10)
