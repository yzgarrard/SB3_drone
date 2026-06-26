import argparse
import os
import time

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

import custom_envs


def quat_conj(quat: np.ndarray) -> np.ndarray:
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def normalize(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return fallback.astype(np.float64)
    return vec / norm


def yaw_from_quat(quat: np.ndarray) -> float:
    return float(R.from_quat(quat, scalar_first=True).as_euler("zyx", degrees=False)[0])


def wrap_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class PositionPIDController:
    def __init__(
        self,
        dt: float,
        mass: float,
        max_thrust: float,
        gravity: float = 9.81,
        kp: np.ndarray | None = None,
        ki: np.ndarray | None = None,
        kd: np.ndarray | None = None,
        integral_limit: np.ndarray | None = None,
        accel_limit: np.ndarray | None = None,
        attitude_gain: np.ndarray | None = None,
        yaw_sp: float = 0.0,
    ):
        self.dt = dt
        self.mass = mass
        self.max_thrust = max_thrust
        self.gravity = gravity
        self.kp = np.array([1.2, 1.2, 2.0], dtype=np.float64) if kp is None else kp
        self.ki = np.array([0.0, 0.0, 0.15], dtype=np.float64) if ki is None else ki
        self.kd = np.array([1.4, 1.4, 2.0], dtype=np.float64) if kd is None else kd
        self.integral_limit = (
            np.array([1.0, 1.0, 1.5], dtype=np.float64)
            if integral_limit is None
            else integral_limit
        )
        self.accel_limit = (
            np.array([3.0, 3.0, 5.0], dtype=np.float64)
            if accel_limit is None
            else accel_limit
        )
        self.attitude_gain = (
            np.array([4.0, 4.0, 2.5], dtype=np.float64)
            if attitude_gain is None
            else attitude_gain
        )
        self.yaw_sp = yaw_sp
        self.integral = np.zeros(3, dtype=np.float64)

    def reset(self) -> None:
        self.integral[:] = 0.0

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        pos_err = obs[0:3].astype(np.float64)
        quat = obs[3:7].astype(np.float64)
        vel = obs[7:10].astype(np.float64)

        self.integral = np.clip(
            self.integral + pos_err * self.dt,
            -self.integral_limit,
            self.integral_limit,
        )
        acc_cmd = self.kp * pos_err + self.ki * self.integral - self.kd * vel
        acc_cmd = np.clip(acc_cmd, -self.accel_limit, self.accel_limit)

        force_vec = np.array(
            [acc_cmd[0], acc_cmd[1], self.gravity + acc_cmd[2]],
            dtype=np.float64,
        )
        thrust = self.mass * np.linalg.norm(force_vec)
        body_z_sp = normalize(force_vec, np.array([0.0, 0.0, 1.0]))

        y_c = np.array([-np.sin(self.yaw_sp), np.cos(self.yaw_sp), 0.0])
        body_x_sp = normalize(np.cross(y_c, body_z_sp), np.array([1.0, 0.0, 0.0]))
        body_y_sp = normalize(np.cross(body_z_sp, body_x_sp), np.array([0.0, 1.0, 0.0]))
        rot_sp = np.column_stack([body_x_sp, body_y_sp, body_z_sp])
        quat_sp = R.from_matrix(rot_sp).as_quat(scalar_first=True)

        q_err = quat_mul(quat_conj(quat), quat_sp)
        if q_err[0] < 0.0:
            q_err = -q_err

        rate_sp = 2.0 * self.attitude_gain * q_err[1:4]

        action = np.empty(4, dtype=np.float32)
        action[0] = 2.0 * thrust / self.max_thrust - 1.0
        action[1:] = rate_sp
        return np.clip(action, -1.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a position PID controller on TacDrone v5.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Run as fast as possible instead of matching the MuJoCo timestep.",
    )
    parser.add_argument(
        "--hold-final",
        type=float,
        default=3.0,
        help="Seconds to keep the human viewer open after the last episode.",
    )
    parser.add_argument("--print-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_mode = None if args.no_render else "human"
    env = gym.make("custom_envs/TacDroneHover-v5", render_mode=render_mode)
    controller_dt = float(env.unwrapped.dt)
    use_realtime = render_mode is not None and not args.no_realtime

    try:
        controller = PositionPIDController(
            dt=controller_dt,
            mass=float(env.unwrapped.model.body_mass.sum()),
            max_thrust=float(env.unwrapped.max_thrust),
        )

        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            controller.reset()
            terminated = False
            truncated = False
            step = 0
            info = {}

            while not (terminated or truncated):
                step_started_at = time.perf_counter()
                action = controller.compute_action(obs)
                obs, _reward, terminated, truncated, info = env.step(action)
                if render_mode is not None:
                    env.render()
                if use_realtime:
                    elapsed = time.perf_counter() - step_started_at
                    time.sleep(max(0.0, controller_dt - elapsed))

                if step % args.print_every == 0 or terminated or truncated:
                    pos_err_norm = float(np.linalg.norm(obs[0:3]))
                    yaw_err = wrap_pi(yaw_from_quat(obs[3:7]) - controller.yaw_sp)
                    termination_reason = info.get("termination_reason")
                    print(
                        f"episode={episode} step={step:04d} "
                        f"pos_err={pos_err_norm:.4f} yaw_err={yaw_err:.4f} "
                        f"tilt_deg={info.get('tilt_deg', float('nan')):.2f} "
                        f"action={np.array2string(action, precision=3)} "
                        f"termination={termination_reason}",
                        flush=True,
                    )
                step += 1

        if render_mode is not None and args.hold_final > 0.0:
            time.sleep(args.hold_final)
    finally:
        if render_mode is None:
            env.close()


if __name__ == "__main__":
    main()
