import os

import gymnasium
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines.mad_pod_racing import MapPodRacing
from stable_baselines3.common.env_util import make_vec_env

# env = gymnasium.make("CartPole-v1")

env = make_vec_env(lambda: MapPodRacing(), n_envs=1)

save_dir = "./runs/"
os.makedirs(save_dir, exist_ok=True)

policy_kwargs = dict(activation_fn=torch.nn.Tanh,
                     net_arch=dict(pi=[128, 128], vf=[128, 128]))
model = PPO(MlpPolicy, env, verbose=0, tensorboard_log=save_dir, ent_coef=0.01, learning_rate=10e-4,
            policy_kwargs=policy_kwargs)

model.load("./runs/PPO_20/model")

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, warn=False)

print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Train the agent
model.learn(total_timesteps=5_000_000)

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# save model
model.save(save_dir + "model")
