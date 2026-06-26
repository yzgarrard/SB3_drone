import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure


tmp_path = f"runs/test_Pendulum_v00_Gaussian"
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

env = gym.make("Pendulum-v1")

model = PPO("MlpPolicy", env, verbose=1, device="cpu", seed=0)
model.set_logger(new_logger)
model.learn(total_timesteps=1000000)

# eval_env = gym.make("Pendulum-v1", render_mode="human")
# obs, info = eval_env.reset()

# for _ in range(1000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = eval_env.step(action)

#     if terminated or truncated:
#         obs, info = eval_env.reset()

# eval_env.close()
env.close()
