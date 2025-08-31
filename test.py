import gymnasium as gym
import imageio

from mad_pod_racing import MapPodRacing, supervised_action_choose

gym.register(
        id="gymnasium_env/MapPodRacing-v0",
        entry_point=MapPodRacing,
        max_episode_steps=2000,  # Prevent infinite episodes
    )
env = gym.make("gymnasium_env/MapPodRacing-v0")


score = 0
done = False
observation, info = env.reset()
frames = [env.render()]
while not done:
    action = supervised_action_choose(observation)
    observation_, reward, done, truncated, info = env.step(action)
    score += reward
    frames.append(env.render())
    observation = observation_
print("score ", str(score))
imageio.mimsave("mad_pod_episode.gif", frames, fps=10)