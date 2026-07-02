# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from .rsl_rl_ppo_cfg import PPORunnerCfg as BasePPORunnerCfg


@configclass
class AlternatingWalk_PPORunnerCfg(BasePPORunnerCfg):
    pass
