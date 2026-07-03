# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import PPORunnerCfg as BasePPORunnerCfg


@configclass
class Terrian_PPORunnerCfg(BasePPORunnerCfg):
    experiment_name = "wheelleg_terrian"
    max_iterations = 2000
