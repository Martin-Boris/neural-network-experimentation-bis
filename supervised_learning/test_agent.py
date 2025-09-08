import gymnasium as gym
import imageio
import torch
import torch.nn as nn
from supervised_learning.mad_pod_racing import MapPodRacing, supervised_action_choose
from supervised_learning.main import Agent

gym.register(
        id="gymnasium_env/MapPodRacing-v0",
        entry_point=MapPodRacing,
    )
env = gym.make("gymnasium_env/MapPodRacing-v0")


score = 0
done = False
truncated = False
observation, info = env.reset()
print(observation)
frames = [env.render()]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = Agent(input_dim=env.observation_space.shape[0], output_dim=env.action_space.shape[0]).to(device)
agent.load_checkpoint()
while not done and not truncated:
    action = agent.get_action_and_value(torch.from_numpy(observation).to(device=device))

    observation_, reward, done, truncated, info = env.step(action.detach())
    score += reward
    frames.append(env.render())
    observation = observation_
print("score ", str(score))
imageio.mimsave("mad_pod_episode.gif", frames, fps=10)