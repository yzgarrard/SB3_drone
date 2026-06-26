import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import time

import gymnasium as gym
import custom_envs

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

seed = 0
train_length = 1000000

log_path = (
    f"runs/drone_20260625_unsquashed_v5/test_Drone_PPO_v00_"
    f"_{time.strftime('%Y%m%d_%H%M%S')}_"
    f"_seed_{seed:04d}_"
    f"_Gaussian_"
)
new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])

env = gym.make("custom_envs/TacDroneHover-v5")

model = PPO("MlpPolicy", 
            env, 
            verbose=0, 
            device="cpu", 
            seed=seed)
model.set_logger(new_logger)
model.learn(total_timesteps=train_length)

# eval_env = gym.make("custom_envs/TacDroneHover-v5")
# obs, info = eval_env.reset()

# for _ in range(1000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = eval_env.step(action)

#     if terminated or truncated:
#         obs, info = eval_env.reset()

# eval_env.close()
env.close()
