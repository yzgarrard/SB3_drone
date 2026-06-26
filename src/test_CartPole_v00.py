import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

tmp_path = "runs/test_CartPole_v00_gSDE"
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

env = gym.make("CartPole-v1")

model = PPO("MlpPolicy", env, verbose=1, device="cpu")
model.set_logger(new_logger)
model.learn(total_timesteps=100000)

eval_env = gym.make("CartPole-v1", render_mode="human")
obs, info = eval_env.reset()

for _ in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)

    if terminated or truncated:
        obs, info = eval_env.reset()

eval_env.close()
env.close()
