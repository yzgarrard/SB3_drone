import os

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

_DEFAULT_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tacdrone.xml")


def _wrap_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class TacDroneHoverEnvV08(gym.Env):
    """
    ## Observation Space  (Box, shape=(17,), dtype=float32)
    Index | Quantity
    ------|----------------------------------------------------------
    0-2   | Position error ex, ey, ez  (world frame)
    3-6   | Quaternion  qw, qx, qy, qz
    7-9   | Linear velocity  vx, vy, vz  (world frame)
    10-12 | Angular velocity  wx, wy, wz  (body frame, gyro)
    13-16 | Previous action

    ## Action Space  (Box, shape=(4,), dtype=float32)
    Normalized collective thrust plus body-rate setpoints in [-1, 1].
    action[0] maps to total thrust in [0, max_thrust].
    action[1:4] are roll, pitch, yaw body-rate setpoints in rad/s.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str = _DEFAULT_XML,
        render_mode: str | None = None,
        max_episode_steps: int = 1000,
    ):
        super().__init__()

        self.xml_path = xml_path
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps

        self.model = mujoco.MjModel.from_xml_path(xml_path)  # pyright: ignore[reportAttributeAccessIssue]
        self.data = mujoco.MjData(self.model)  # pyright: ignore[reportAttributeAccessIssue]

        self.frame_skip = 1
        self.max_thrust = 40.0
        self.dt = self.model.opt.timestep * self.frame_skip

        self.MC_ROLLRATE_P = 0.15
        self.MC_ROLLRATE_D = 0.003
        self.MC_PITCHRATE_P = 0.15
        self.MC_PITCHRATE_D = 0.003
        self.MC_YAWRATE_P = 0.2
        self.MC_YAWRATE_D = 0.0

        s45 = np.sin(np.deg2rad(45.0))
        d = 0.225
        k_tau = 0.0167
        self.CAM = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [-d * s45, d * s45, d * s45, -d * s45],
                [-d * s45, d * s45, -d * s45, d * s45],
                [-k_tau, -k_tau, k_tau, k_tau],
            ],
            dtype=np.float64,
        )
        self.CAM_inv = np.linalg.inv(self.CAM)

        self.motor_time_constant = 0.059
        self.alpha = np.exp(-self.dt / self.motor_time_constant)

        self.observation_space = spaces.Box(
            low=np.full(17, -np.inf, dtype=np.float32),
            high=np.full(17, np.inf, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-np.ones(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            dtype=np.float32,
        )

        self.pos_des = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(4, dtype=np.float32)

        self._viewer = None
        self._viewer_launched = False
        self._renderer = None
        self._step_count = 0
        self._init_xy = np.zeros(2, dtype=np.float64)

    def _get_obs(self) -> np.ndarray:
        pos = self.data.qpos[:3].copy()
        quat = self.data.qpos[3:7].copy()
        vel = self.data.qvel[:3].copy()
        gyro = self.data.sensor("body_gyro").data.copy()
        pos_err = (self.pos_des - pos).astype(np.float32)
        return np.concatenate([pos_err, quat, vel, gyro, self.last_action]).astype(np.float32)

    def _tilt_angle(self) -> float:
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, self.data.qpos[3:7])  # pyright: ignore[reportAttributeAccessIssue]
        body_z = rot.reshape(3, 3)[:, 2]
        cos_tilt = float(np.clip(body_z[2], -1.0, 1.0))
        return float(np.arccos(cos_tilt))

    def _yaw(self) -> float:
        quat = self.data.qpos[3:7]
        return float(R.from_quat(quat, scalar_first=True).as_euler("zyx", degrees=False)[0])

    def _compute_reward(self, action_normed: np.ndarray) -> tuple[float, dict[str, float]]:
        pos_err = (self.pos_des - self.data.qpos[:3]).astype(np.float32)
        vel = self.data.qvel[:3]
        gyro = self.data.sensor("body_gyro").data
        tilt = self._tilt_angle()
        yaw_err = _wrap_pi(self._yaw())
        action_delta = action_normed - self.last_action

        pos_norm_sq = float(np.dot(pos_err, pos_err))
        vel_norm_sq = float(np.dot(vel, vel))
        gyro_norm_sq = float(np.dot(gyro, gyro))
        hover_ready = (
            np.linalg.norm(pos_err) < 0.10
            and np.linalg.norm(vel) < 0.20
            and tilt < np.deg2rad(5.0)
        )

        reward_terms = {
            "alive": 1.0,
            "pos": float(2.0 * np.exp(-4.0 * pos_norm_sq)),
            "vel": float(0.5 * np.exp(-0.25 * vel_norm_sq)),
            "tilt": float(0.3 * np.exp(-8.0 * tilt**2)),
            "yaw": float(0.1 * np.exp(-2.0 * yaw_err**2)),
            "hover_bonus": 1.0 if hover_ready else 0.0,
            "act_delta": float(-0.01 * np.dot(action_delta, action_delta)),
        }
        reward_terms["ang"] = float(-0.05 * gyro_norm_sq)
        reward_terms["total"] = float(sum(reward_terms.values()))
        return reward_terms["total"], reward_terms

    def _termination_reason(self) -> str | None:
        pos = self.data.qpos[:3]
        tilt = self._tilt_angle()
        if pos[2] < 0.05:
            return "ground_contact"
        if abs(pos[0]) > 5.0 or abs(pos[1]) > 5.0:
            return "xy_bounds"
        if tilt > np.deg2rad(60.0):
            return "excessive_tilt"
        return None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

        rng = self.np_random
        self.data.qpos[0] = rng.uniform(-0.5, 0.5)
        self.data.qpos[1] = rng.uniform(-0.5, 0.5)
        self.data.qpos[2] = 0.135 + rng.uniform(0.0, 1.0)

        self.pos_des[0:2] = rng.uniform(-0.5, 0.5, size=2)
        self.pos_des[2] = rng.uniform(0.5, 2.0)
        self.last_action = np.zeros(4, dtype=np.float32)

        mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
        self._init_xy = self.data.qpos[:2].copy()
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action_normed = np.clip(action, -1.0, 1.0).astype(np.float32)

        thrust_sp = (action_normed[0] + 1.0) * 0.5 * self.max_thrust
        omega_sp = action_normed[1:]
        omega = self.data.sensor("body_gyro").data.copy()
        omega_err = omega_sp - omega

        tau_sp = np.array(
            [
                self.MC_ROLLRATE_P * omega_err[0] - self.MC_ROLLRATE_D * omega[0],
                self.MC_PITCHRATE_P * omega_err[1] - self.MC_PITCHRATE_D * omega[1],
                self.MC_YAWRATE_P * omega_err[2] - self.MC_YAWRATE_D * omega[2],
            ],
            dtype=np.float64,
        )
        motor_force_cmd = self.CAM_inv @ np.concatenate([[thrust_sp], tau_sp])
        motor_force_cmd = np.clip(motor_force_cmd, 0.0, self.max_thrust / 4.0)

        for _ in range(self.frame_skip):
            self.data.ctrl[:] = self.alpha * self.data.ctrl[:] + (1.0 - self.alpha) * motor_force_cmd
            mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

        reward, reward_terms = self._compute_reward(action_normed)
        termination_reason = self._termination_reason()
        terminated = termination_reason is not None
        if terminated:
            reward_terms["termination"] = -300.0
            reward = float(reward + reward_terms["termination"])
            reward_terms["total"] = reward

        self._step_count += 1
        truncated = self._step_count >= self.max_episode_steps
        if truncated and termination_reason is None:
            termination_reason = "time_limit"
        self.last_action = action_normed.copy()
        obs = self._get_obs()

        info = {
            "z": float(self.data.qpos[2]),
            "z_err": float(self.pos_des[2] - self.data.qpos[2]),
            "tilt_deg": float(np.rad2deg(self._tilt_angle())),
            "yaw": self._yaw(),
            "x_err": float(self.pos_des[0] - self.data.qpos[0]),
            "y_err": float(self.pos_des[1] - self.data.qpos[1]),
            "pos_err_norm": float(np.linalg.norm(obs[0:3])),
            "pos_des": self.pos_des.copy(),
            "reward_terms": reward_terms,
            "termination_reason": termination_reason,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if not self._viewer_launched:
                import mujoco.viewer as mjv

                self._viewer = mjv.launch_passive(self.model, self.data)
                self._viewer_launched = True

            if self._viewer is not None and self._viewer.is_running():
                self._viewer.sync()

        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data, camera="track")
            return self._renderer.render()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self._viewer_launched = False
        if self._renderer is not None:
            self._renderer.close()  # pyright: ignore[reportAttributeAccessIssue]
            self._renderer = None
