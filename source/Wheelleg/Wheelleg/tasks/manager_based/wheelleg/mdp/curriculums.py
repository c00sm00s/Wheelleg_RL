# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum helpers for WheelLeg tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _as_env_ids_tensor(env: ManagerBasedRLEnv, env_ids: Sequence[int] | slice | None) -> torch.Tensor:
    """Return environment ids as a tensor on the env device."""

    if env_ids is None or env_ids == slice(None):
        return torch.arange(env.num_envs, device=env.device)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.tensor(env_ids, dtype=torch.long, device=env.device)


def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Four-stage terrain curriculum based on episode timeout success.

    Stage 0 uses row 0 / column 0 (flat), stage 1 uses row 0 / column 1
    (rough), stage 2 uses row 0 / column 2 (down stairs), and stage 3 uses
    row 0 / column 3 (up stairs). An environment moves up one stage only when
    the previous episode ended by ``time_out``. If it terminates early, it moves
    down one stage to relearn on easier ground.
    """

    terrain: TerrainImporter = env.scene.terrain
    if terrain.terrain_origins is None:
        return torch.zeros((), device=env.device)

    ids = _as_env_ids_tensor(env, env_ids)
    _, num_cols = terrain.terrain_origins.shape[:2]
    max_stage = min(num_cols, 4) - 1

    # During the very first env.reset(), ManagerBasedRLEnv has not stepped yet,
    # so these buffers do not exist. Start every env on flat stage 0.
    if not hasattr(env, "reset_time_outs") or not hasattr(env, "reset_terminated"):
        stage = torch.zeros_like(ids)
    else:
        move_up = env.reset_time_outs[ids]
        move_down = env.reset_terminated[ids] & ~move_up
        stage = terrain.terrain_types[ids] + 1 * move_up - 1 * move_down
        stage = torch.clamp(stage, min=0, max=max_stage)

    # Keep row fixed at 0 and use the column as the course stage:
    # col 0 -> flat, col 1 -> rough, col 2 -> down stairs, col 3 -> up stairs.
    terrain.terrain_levels[ids] = 0
    terrain.terrain_types[ids] = stage
    terrain.env_origins[ids] = terrain.terrain_origins[0, stage]

    return torch.mean(stage.float())


def terrian_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Backward-compatible alias for the old misspelled helper name."""

    return terrain_levels_vel(env, env_ids, asset_cfg)


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy_exp",
    increment: float = 0.1,
    success_ratio: float = 0.8,
) -> torch.Tensor:
    """Expand XY velocity command range after tracking becomes reliable.

    The terrain curriculum teaches harder terrain rows. This command curriculum
    teaches faster locomotion by widening the current command range toward the
    ``limit_ranges`` stored on ``UniformLevelVelocityCommandCfg``.
    """

    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    episode_reward = env.reward_manager._episode_sums[reward_term_name][env_ids]
    reward = torch.mean(episode_reward) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0 and reward > reward_term.weight * success_ratio:
        delta = torch.tensor([-increment, increment], device=env.device)
        ranges.lin_vel_x = torch.clamp(
            torch.tensor(ranges.lin_vel_x, device=env.device) + delta,
            limit_ranges.lin_vel_x[0],
            limit_ranges.lin_vel_x[1],
        ).tolist()
        ranges.lin_vel_y = torch.clamp(
            torch.tensor(ranges.lin_vel_y, device=env.device) + delta,
            limit_ranges.lin_vel_y[0],
            limit_ranges.lin_vel_y[1],
        ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z_exp",
    increment: float = 0.1,
    success_ratio: float = 0.8,
) -> torch.Tensor:
    """Expand yaw-rate command range after yaw tracking becomes reliable."""

    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    episode_reward = env.reward_manager._episode_sums[reward_term_name][env_ids]
    reward = torch.mean(episode_reward) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0 and reward > reward_term.weight * success_ratio:
        delta = torch.tensor([-increment, increment], device=env.device)
        ranges.ang_vel_z = torch.clamp(
            torch.tensor(ranges.ang_vel_z, device=env.device) + delta,
            limit_ranges.ang_vel_z[0],
            limit_ranges.ang_vel_z[1],
        ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)
