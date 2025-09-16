import gym
import numpy as np


class SelfPlayWrapper(gym.Env):
    def __init__(self, env, opponent_policy):
        self.env = env
        self.opponent_policy = opponent_policy

        # Assuming both agents have same observation and action space
        self.observation_space = env.observation_space['player_1']
        self.action_space = env.action_space['player_1']

    def reset(self):
        obs = self.env.reset()
        return obs['player_1']

    def step(self, action):
        # Get the opponent's action
        opponent_obs = self.env.get_obs()['player_2']
        opponent_action = self.opponent_policy.predict(opponent_obs, deterministic=True)[0]

        # Step the environment with both actions
        actions = {
            'player_1': action,
            'player_2': opponent_action
        }

        obs, rewards, done, _, info = self.env.step(actions)
        return obs['player_1'], rewards['player_1'], done, info
