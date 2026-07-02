# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum helpers for WheelLeg tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
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


def _ensure_path_length_buffers(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> None:
    """Create per-env path length buffers if this is the first call."""

    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "wheelleg_path_length"):
        env.wheelleg_path_length = torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "wheelleg_prev_root_xy"):
        env.wheelleg_prev_root_xy = asset.data.root_pos_w[:, :2].clone()


def reset_path_length(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | slice | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset the per-episode path-length odometer after root-state reset."""

    asset: Articulation = env.scene[asset_cfg.name]
    _ensure_path_length_buffers(env, asset_cfg)
    ids = _as_env_ids_tensor(env, env_ids)
    env.wheelleg_path_length[ids] = 0.0
    env.wheelleg_prev_root_xy[ids] = asset.data.root_pos_w[ids, :2]


def update_path_length(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | slice | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Accumulate scalar XY path length traveled during the episode."""

    asset: Articulation = env.scene[asset_cfg.name]
    _ensure_path_length_buffers(env, asset_cfg)
    ids = _as_env_ids_tensor(env, env_ids)
    current_xy = asset.data.root_pos_w[ids, :2]
    step_distance = torch.norm(current_xy - env.wheelleg_prev_root_xy[ids], dim=1)
    env.wheelleg_path_length[ids] += step_distance
    env.wheelleg_prev_root_xy[ids] = current_xy


def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Three-stage terrain curriculum using scalar traveled path length.

    Stage 0 uses terrain column 0 (flat), stage 1 uses column 1 (rough), and
    stage 2 uses column 2 (small stairs). Successful envs move up one stage
    when their accumulated XY path length passes the threshold; envs that travel
    less than half of commanded path length move down one stage.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    if terrain.terrain_origins is None:
        return torch.zeros((), device=env.device)

    if not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, dtype=torch.long, device=env.device)

    num_rows, num_cols = terrain.terrain_origins.shape[:2]
    max_stage = min(num_rows, num_cols, 3) - 1

    _ensure_path_length_buffers(env, asset_cfg)
    path_length = env.wheelleg_path_length[env_ids]
    move_up = path_length > terrain.cfg.terrain_generator.size[0] / 2
    move_down = path_length < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up

    stage = terrain.terrain_levels[env_ids] + 1 * move_up - 1 * move_down
    stage = torch.clamp(stage, min=0, max=max_stage)

    # Keep row and column equal to the stage index:
    # 0 -> flat, 1 -> rough, 2 -> small stairs.
    terrain.terrain_levels[env_ids] = stage
    terrain.terrain_types[env_ids] = stage
    terrain.env_origins[env_ids] = terrain.terrain_origins[stage, stage]

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
