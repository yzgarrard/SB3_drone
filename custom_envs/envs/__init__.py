from gymnasium.envs.registration import register
from custom_envs.envs.dji_f450 import DJIF450EnvV0p0
from custom_envs.envs.dji_f450_res_acc import DJIF450EnvV1p0
from custom_envs.envs.tacdrone_env import TacDroneHoverEnv
from custom_envs.envs.tacdrone_env_v02 import TacDroneHoverEnvV02
from custom_envs.envs.tacdrone_env_v03 import TacDroneHoverEnvV03
from custom_envs.envs.tacdrone_env_v04 import TacDroneHoverEnvV04
from custom_envs.envs.tacdrone_env_v05 import TacDroneHoverEnvV05
from custom_envs.envs.tacdrone_env_v06 import TacDroneHoverEnvV06
from custom_envs.envs.tacdrone_env_v07 import TacDroneHoverEnvV07
from custom_envs.envs.tacdrone_env_v08 import TacDroneHoverEnvV08

register(
    id="custom_envs/DJIF450-v0",
    entry_point="custom_envs.envs:DJIF450EnvV0p0",
    max_episode_steps=1000,
)

register(
    id="custom_envs/DJIF450-v1",
    entry_point="custom_envs.envs:DJIF450EnvV1p0",
    max_episode_steps=1000,
)

register(
    id="custom_envs/TacDroneHover-v0",
    entry_point="custom_envs.envs:TacDroneHoverEnv",
    max_episode_steps=1000,
)

register(
    id="custom_envs/TacDroneHover-v2",
    entry_point="custom_envs.envs:TacDroneHoverEnvV02",
    max_episode_steps=1000,
)

register(
    id="custom_envs/TacDroneHover-v3",
    entry_point="custom_envs.envs:TacDroneHoverEnvV03",
    max_episode_steps=1000,
)

register(
    id="custom_envs/TacDroneHover-v4",
    entry_point="custom_envs.envs:TacDroneHoverEnvV04",
    max_episode_steps=1000,
)

register(
    id="custom_envs/TacDroneHover-v5",
    entry_point="custom_envs.envs:TacDroneHoverEnvV05",
    max_episode_steps=1000,
)

register(
    id="custom_envs/TacDroneHover-v6",
    entry_point="custom_envs.envs:TacDroneHoverEnvV06",
    max_episode_steps=1000,
)

register(
    id="custom_envs/TacDroneHover-v7",
    entry_point="custom_envs.envs:TacDroneHoverEnvV07",
    max_episode_steps=1000,
)

register(
    id="custom_envs/TacDroneHover-v8",
    entry_point="custom_envs.envs:TacDroneHoverEnvV08",
    max_episode_steps=1000,
)
