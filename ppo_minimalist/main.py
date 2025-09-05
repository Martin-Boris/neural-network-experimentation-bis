import gymnasium as gym
import numpy as np

from ppo_minimalist.env.mad_pod_racing import MapPodRacing
from ppo_torch import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    # env = gym.make('CartPole-v1')
    gym.register(
        id="gymnasium_env/MapPodRacing-v0",
        entry_point=MapPodRacing
    )
    env = gym.make("gymnasium_env/MapPodRacing-v0")
    N = 64
    batch_size = 32
    n_epochs = 3
    alpha = 0.0003
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs,
                  input_dims=env.observation_space.shape, policy_clip=0.2)
    n_games = 1000

    figure_file = 'plots/cartpole.png'

    best_score = 0
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation, _ = env.reset()
        done = False
        truncated = False
        score = 0
        completion_cp = 0
        while not done and not truncated:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, truncated, info = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        completion_cp = info['cp_completion']
        if avg_score > best_score:
            best_score = avg_score
            # agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_steps', n_steps, 'learning_steps', learn_iters, 'completion_cp', completion_cp)
    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
    agent.save_models()
