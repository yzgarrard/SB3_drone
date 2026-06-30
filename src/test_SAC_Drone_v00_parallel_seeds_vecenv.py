import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import concurrent.futures
import multiprocessing
import sys
import time

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import gymnasium as gym
import custom_envs
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

SEED_LIST = [1,2,3,4]
TOTAL_TIMESTEPS = 1_000_000
N_ENVS = 1
learning_rate = 1e-4


def train_seed(seed: int) -> dict:
    run_path = (
        f"runs/drone_20260629_0_SAC_vecenv_normobs_nenvs_{N_ENVS}/"
        f"_{time.strftime('%Y%m%d_%H%M%S')}_"
        f"_seed_{seed:04d}_"
        f"_Gaussian_"
    )
    start_time = time.perf_counter()
    env = None
    eval_env = None

    try:
        new_logger = configure(run_path, ["csv", "tensorboard"])
        env = DummyVecEnv([lambda: gym.make("custom_envs/TacDroneHover-v8") for _ in range(N_ENVS)])
        env = VecMonitor(env)
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        eval_env = DummyVecEnv([lambda: gym.make("custom_envs/TacDroneHover-v8")])
        eval_env = VecMonitor(eval_env)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
        eval_env.seed(123456789)
        eval_callback = EvalCallback(
            eval_env, 
            n_eval_episodes=50, 
            eval_freq=max(50000 // N_ENVS, 1),
            best_model_save_path=run_path,
            verbose=0)

        policy_kwargs = dict(net_arch=dict(pi=[64, 64], qf=[64, 64]))

        model = SAC(
            "MlpPolicy",
            env,
            learning_starts=100000,
            learning_rate=learning_rate,
            train_freq=1,
            gradient_steps=1,
            batch_size=256,
            gamma=0.99,
            verbose=0,
            device="cpu",
            seed=seed,
            policy_kwargs=policy_kwargs,
        )
        model.set_logger(new_logger)
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)

        return {
            "seed": seed,
            "run_path": run_path,
            "total_timesteps": TOTAL_TIMESTEPS,
            "elapsed_sec": time.perf_counter() - start_time,
            "success": True,
        }
    finally:
        if env is not None:
            env.close()
        if eval_env is not None:
            eval_env.close()


def main() -> None:
    print(f"Starting {len(SEED_LIST)} Gaussian training jobs.")
    print(f"Timesteps per job: {TOTAL_TIMESTEPS}")
    print(f"Seeds: {SEED_LIST}")

    failures = []
    results = []
    context = multiprocessing.get_context("spawn")

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=len(SEED_LIST),
        mp_context=context,
    ) as executor:
        future_to_seed = {executor.submit(train_seed, seed): seed for seed in SEED_LIST}

        for seed in SEED_LIST:
            print(f"Submitted seed={seed}")

        for future in concurrent.futures.as_completed(future_to_seed):
            seed = future_to_seed[future]
            try:
                result = future.result()
            except Exception as exc:
                failures.append(seed)
                print(f"Failed seed={seed}: {exc!r}")
            else:
                results.append(result)
                print(
                    "Finished "
                    f"seed={result['seed']} "
                    f"elapsed={result['elapsed_sec']:.1f}s "
                    f"run_path={result['run_path']}"
                )

    if failures:
        print(f"Failed seeds: {sorted(failures)}")
        print(f"Completed seeds: {sorted(result['seed'] for result in results)}")
        sys.exit(1)

    print("All Gaussian training jobs completed.")


if __name__ == "__main__":
    main()
