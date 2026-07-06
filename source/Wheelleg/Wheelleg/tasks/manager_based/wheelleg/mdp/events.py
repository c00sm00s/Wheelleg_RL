# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific reset events for WheelLeg terrain locomotion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_root_state_uniform_stage_yaw(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    stage: int,
    stage_yaw_range: tuple[float, float],
    stage_pose_ranges: dict[int, dict[str, tuple[float, float]]] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset root state uniformly, with optional stage-specific pose ranges."""

    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    terrain = getattr(env.scene, "terrain", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    if terrain_types is not None:
        if stage_pose_ranges is not None:
            key_to_index = {"x": 0, "y": 1, "z": 2, "roll": 3, "pitch": 4, "yaw": 5}
            for stage_id, stage_pose_range in stage_pose_ranges.items():
                stage_mask = terrain_types[env_ids] == int(stage_id)
                if not torch.any(stage_mask):
                    continue
                for key, value_range in stage_pose_range.items():
                    rand_index = key_to_index.get(key)
                    if rand_index is None:
                        continue
                    values = torch.empty(
                        int(torch.count_nonzero(stage_mask).item()),
                        dtype=rand_samples.dtype,
                        device=asset.device,
                    )
                    rand_samples[stage_mask, rand_index] = values.uniform_(*value_range)
        stage_mask = terrain_types[env_ids] == stage
        if torch.any(stage_mask):
            stage_yaw = torch.empty(int(torch.count_nonzero(stage_mask).item()), dtype=rand_samples.dtype, device=asset.device)
            rand_samples[stage_mask, 5] = stage_yaw.uniform_(*stage_yaw_range)

    origin_source = env.scene.env_origins
    terrain = getattr(env.scene, "terrain", None)
    terrain_origins = getattr(terrain, "env_origins", None)
    if terrain_origins is not None:
        origin_source = terrain_origins

    positions = root_states[:, 0:3] + origin_source[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    velocities = root_states[:, 7:13] + rand_samples

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
