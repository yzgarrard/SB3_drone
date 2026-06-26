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

FREQ_LIST = [-1, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
TOTAL_TIMESTEPS = 1_000_000


def train_freq(freq: int) -> dict:
    run_path = f"runs/test_Pendulum_v01_gSDE_sample_freq_{freq:04d}"
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
            sde_sample_freq=freq,
            seed=0,
        )
        model.set_logger(new_logger)
        model.learn(total_timesteps=TOTAL_TIMESTEPS)

        return {
            "freq": freq,
            "run_path": run_path,
            "total_timesteps": TOTAL_TIMESTEPS,
            "elapsed_sec": time.perf_counter() - start_time,
            "success": True,
        }
    finally:
        if env is not None:
            env.close()


def main() -> None:
    print(f"Starting {len(FREQ_LIST)} gSDE training jobs.")
    print(f"Timesteps per job: {TOTAL_TIMESTEPS}")
    print(f"Frequencies: {FREQ_LIST}")

    failures = []
    results = []
    context = multiprocessing.get_context("spawn")

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=len(FREQ_LIST),
        mp_context=context,
    ) as executor:
        future_to_freq = {executor.submit(train_freq, freq): freq for freq in FREQ_LIST}

        for freq in FREQ_LIST:
            print(f"Submitted freq={freq}")

        for future in concurrent.futures.as_completed(future_to_freq):
            freq = future_to_freq[future]
            try:
                result = future.result()
            except Exception as exc:
                failures.append(freq)
                print(f"Failed freq={freq}: {exc!r}")
            else:
                results.append(result)
                print(
                    "Finished "
                    f"freq={result['freq']} "
                    f"elapsed={result['elapsed_sec']:.1f}s "
                    f"run_path={result['run_path']}"
                )

    if failures:
        print(f"Failed frequencies: {sorted(failures)}")
        print(f"Completed frequencies: {sorted(result['freq'] for result in results)}")
        sys.exit(1)

    print("All gSDE training jobs completed.")


if __name__ == "__main__":
    main()
