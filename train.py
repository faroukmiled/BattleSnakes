import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gymnasium.wrappers import RecordEpisodeStatistics
from snake_gym import BattlesnakeGym  # Adjust path if needed

# === 1. Setup logging ===
log_dir = "./ppo_battlesnake_logs/"
os.makedirs(log_dir, exist_ok=True)
# === 2. Create and wrap environment ===
raw_env = BattlesnakeGym(map_size=(11, 11), number_of_snakes=1)
env = RecordEpisodeStatistics(raw_env)

# === 3. Optional: Check environment compatibility ===
# check_env(env, warn=True)

# === 4. Create PPO model ===
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    tensorboard_log=log_dir
)

# === 5. Train the model ===
model.learn(total_timesteps=100_000)

# === 6. Save the model ===
model.save("ppo_battlesnake")

print("Training complete. View logs with:")
print("    tensorboard --logdir ./ppo_battlesnake_logs/")

"""model = PPO.load("ppo_battlesnake")
raw_env = BattlesnakeGym(map_size=(11, 11), number_of_snakes=1)
obs,_ = raw_env.reset()
while True:
    action, _ = model.predict(obs)
    obs, rewards, terminated ,_, info = raw_env.step(action)
    raw_env.render("human")"""