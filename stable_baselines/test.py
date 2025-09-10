import gymnasium as gym
import imageio
import numpy as np

from stable_baselines.mad_pod_racing import MapPodRacing, supervised_action_choose

env = MapPodRacing()

score = 0
done = False
truncated = False
observation, info = env.reset()

frames = [env.render()]
while not done:
    print(observation)
    action = supervised_action_choose(observation)
    observation_, reward, done, truncated, info = env.step(np.array([action]))
    score += reward
    frames.append(env.render())
    observation = observation_
print("score ", str(score))
imageio.mimsave("mad_pod_episode.gif", frames, fps=10)
