# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Wheelleg-Template-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheelleg_env_cfg:WheellegEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Wheelleg-AlternatingWalk-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Wheelleg_AlternatingWalk_env_cfg:WheellegAlternatingWalkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_AlternatingWalk_ppo_cfg:AlternatingWalk_PPORunnerCfg",
    },
)

gym.register(
    id="Wheelleg-Jump-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Wheelleg_Jump_env_cfg:WheellegJumpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_Jump_ppo_cfg:Jump_PPORunnerCfg",
    },
)


gym.register(
    id="Wheelleg-Terrian-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Wheelleg_Terrian_env_cfg:WheellegTerrianEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_Terrian_ppo_cfg:Terrian_PPORunnerCfg",
    },
)
