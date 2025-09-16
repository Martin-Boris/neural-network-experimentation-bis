import gymnasium as gym
import imageio
import numpy as np

from stable_baselines.mad_pod_racing import MapPodRacing, supervised_action_choose
from stable_baselines3 import PPO
import torch

env = MapPodRacing()

score = 0
done = False
truncated = False
observation, info = env.reset(seed=1)

frames = [env.render()]
while not done:
    print(observation)
    action = supervised_action_choose(observation)
    observation_, reward, done, truncated, info = env.step(np.array([action, 10, 0.9]))
    score += reward
    frames.append(env.render())
    observation = observation_
print("score ", str(score))
imageio.mimsave("mad_pod_episode.gif", frames, fps=10)

'''## test using SB3
env = MapPodRacing()
model = PPO.load("./runs/PPO_22/model")

score = 0
done = False
truncated = False
observation, info = env.reset(seed=1)
frames = [env.render()]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

while not done:
    actions, _ = model.predict(observation, deterministic=True)
    print(actions)
    observation_, reward, done, truncated, info = env.step(actions)
    score += reward
    frames.append(env.render())
    observation = observation_
print("score SB3", str(score))
imageio.mimsave("mad_pod_episode_sb3.gif", frames, fps=10)'''
