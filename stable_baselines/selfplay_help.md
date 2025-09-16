````python
# Save the trained agent
model.save("ppo_player_1")

# Load it as the new opponent
opponent_policy = PPO.load("ppo_player_1")
env_against_old_version = SelfPlayWrapper(env, opponent_policy)
model2 = PPO("MlpPolicy", env_against_old_version, verbose=1)
model2.learn(total_timesteps=100_000)
````

4. 🔁 Optionally Automate Self-Play Training

You can alternate training between players and periodically update the opponent with past checkpoints:

````python
# Pseudo-code
for i in range(num_iterations):
    train_agent_against_old_version()
    save_current_agent_as_new_opponent()
````