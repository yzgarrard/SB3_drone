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

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

SEED_LIST = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
TOTAL_TIMESTEPS = 1_000_000


def train_seed(seed: int) -> dict:
    run_path = f"runs/test_Pendulum_v01_gSDE_seed_{seed:04d}_sample_freq_0004"
    start_time = time.perf_counter()
    env = None

    try:
        new_logger = configure(run_path, ["csv", "tensorboard"])
        env = gym.make("Pendulum-v1")

        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            device="cpu",
            use_sde=True,
            sde_sample_freq=4,
            seed=seed,
        )
        model.set_logger(new_logger)
        model.learn(total_timesteps=TOTAL_TIMESTEPS)

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


def main() -> None:
    print(f"Starting {len(SEED_LIST)} gSDE training jobs.")
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

    print("All gSDE training jobs completed.")


if __name__ == "__main__":
    main()
