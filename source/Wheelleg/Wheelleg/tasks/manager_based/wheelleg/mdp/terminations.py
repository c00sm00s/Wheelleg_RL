# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific termination helpers for WheelLeg terrain locomotion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, wrap_to_pi

from .observations import contact_state_obs

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _stage_mask(env: ManagerBasedRLEnv, stage: int) -> torch.Tensor:
    terrain = getattr(env.scene, "terrain", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    if terrain_types is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return terrain_types == stage


def _stage3_relative_root_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    asset: Articulation = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    root_pos_w = asset.data.root_pos_w
    rel_pos = root_pos_w - terrain.env_origins
    return rel_pos[:, 0], rel_pos[:, 1], rel_pos[:, 2]


def _stage3_relative_wheel_pos(
    env: ManagerBasedRLEnv,
    wheel_body_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return left/right wheel body positions relative to each terrain origin."""

    asset: Articulation = env.scene[wheel_body_cfg.name]
    wheel_pos_w = asset.data.body_pos_w[:, wheel_body_cfg.body_ids, :]
    return wheel_pos_w - env.scene.terrain.env_origins.unsqueeze(1)


def _stage3_actual_stair_surface_height(
    sample_x: torch.Tensor,
    bottom_platform_width: float,
    step_height: float,
    step_width: float,
    num_steps: int,
) -> torch.Tensor:
    """Return the physical stair/top-platform height under each x sample."""

    stair_start_x = 0.5 * bottom_platform_width
    stair_end_x = stair_start_x + num_steps * step_width
    progress = torch.clamp(sample_x - stair_start_x, min=0.0, max=num_steps * step_width)
    active_step = torch.clamp(torch.floor(progress / step_width) + 1.0, min=1.0, max=float(num_steps))
    surface_z = torch.where(sample_x < stair_start_x, torch.zeros_like(sample_x), active_step * step_height)
    return torch.where(sample_x > stair_end_x, torch.full_like(sample_x, num_steps * step_height), surface_z)


def _stage3_wheel_stance_steps(
    env: ManagerBasedRLEnv,
    wheel_body_cfg: SceneEntityCfg,
    bottom_platform_width: float,
    step_height: float,
    step_width: float,
    num_steps: int,
    wheel_radius: float,
    stance_height_margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return left/right step indices and whether each wheel is close to that surface."""

    wheel_pos = _stage3_relative_wheel_pos(env, wheel_body_cfg)
    wheel_x = wheel_pos[..., 0]
    wheel_bottom_z = wheel_pos[..., 2] - wheel_radius

    stair_start_x = 0.5 * bottom_platform_width
    progress = torch.clamp(wheel_x - stair_start_x, min=0.0, max=num_steps * step_width)
    wheel_steps = torch.clamp(torch.floor(progress / step_width) + 1.0, min=1.0, max=float(num_steps))
    wheel_steps = torch.where(wheel_x < stair_start_x, torch.zeros_like(wheel_steps), wheel_steps)

    surface_z = wheel_steps * step_height
    wheel_on_surface = torch.abs(wheel_bottom_z - surface_z) <= stance_height_margin
    return wheel_steps, wheel_on_surface


def _stage3_top_reached(
    env: ManagerBasedRLEnv,
    stage: int,
    asset_cfg: SceneEntityCfg,
    bottom_platform_width: float,
    step_width: float,
    num_steps: int,
    stair_width: float,
    fall_margin: float,
    top_success_margin: float,
) -> torch.Tensor:
    stage_gate = _stage_mask(env, stage)
    rel_x, rel_y, _ = _stage3_relative_root_state(env, asset_cfg)
    top_x = 0.5 * bottom_platform_width + num_steps * step_width + top_success_margin
    inside_width = torch.abs(rel_y) <= 0.5 * stair_width + fall_margin
    return stage_gate & inside_width & (rel_x >= top_x)


def stage3_reached_top(
    env: ManagerBasedRLEnv,
    stage: int = 3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    bottom_platform_width: float = 2.0,
    step_width: float = 0.35,
    num_steps: int = 10,
    stair_width: float = 2.0,
    fall_margin: float = 0.35,
    top_success_margin: float = 0.50,
) -> torch.Tensor:
    """Return True when a stage3 robot reaches the top platform target line."""

    return _stage3_top_reached(
        env,
        stage,
        asset_cfg,
        bottom_platform_width,
        step_width,
        num_steps,
        stair_width,
        fall_margin,
        top_success_margin,
    )


def stage3_fell_off_stairs(
    env: ManagerBasedRLEnv,
    stage: int = 3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["L_wheel_link", "R_wheel_link"]),
    bottom_platform_width: float = 2.0,
    step_height: float = 0.10,
    step_width: float = 0.35,
    num_steps: int = 10,
    stair_width: float = 2.0,
    fall_margin: float = 0.35,
    top_success_margin: float = 0.50,
    height_check_start_step: int = 2,
    wheel_radius: float = 0.0625,
    wheel_height_margin: float = 0.03,
    require_both_wheels_below: bool = True,
    enable_height_fall_check: bool = True,
) -> torch.Tensor:
    """Return True when a stage3 robot leaves the stair course.

    Height falling is based on wheel-bottom height, not base height. The base
    can pitch or reach over a stair before the wheels climb it, but a wheel
    bottom below the already-completed stair surface is a better sign that the
    robot has dropped beside/behind the course.
    """

    stage_gate = _stage_mask(env, stage)
    rel_x, rel_y, rel_z = _stage3_relative_root_state(env, asset_cfg)
    reached_top = _stage3_top_reached(
        env,
        stage,
        asset_cfg,
        bottom_platform_width,
        step_width,
        num_steps,
        stair_width,
        fall_margin,
        top_success_margin,
    )

    stair_start_x = 0.5 * bottom_platform_width
    stair_end_x = stair_start_x + num_steps * step_width
    active_stair_x = rel_x >= stair_start_x

    lateral_fall = active_stair_x & (torch.abs(rel_y) > 0.5 * stair_width + fall_margin)
    backed_off_start = rel_x < -0.5 * bottom_platform_width - fall_margin

    wheel_pos = _stage3_relative_wheel_pos(env, wheel_body_cfg)
    wheel_x = wheel_pos[..., 0]
    wheel_bottom_z = wheel_pos[..., 2] - wheel_radius

    progress = torch.clamp(wheel_x - stair_start_x, min=0.0, max=num_steps * step_width)
    # Use completed-step height for each wheel instead of the next riser. This
    # lets a wheel touch a riser and learn to lift without being reset before it
    # has climbed onto that step.
    completed_steps = torch.clamp(torch.floor(progress / step_width), min=0.0, max=float(num_steps))
    expected_surface_z = torch.where(wheel_x < stair_start_x, torch.zeros_like(completed_steps), completed_steps * step_height)
    expected_surface_z = torch.where(
        wheel_x > stair_end_x,
        torch.full_like(completed_steps, num_steps * step_height),
        expected_surface_z,
    )
    height_check_x = stair_start_x + max(height_check_start_step, 0) * step_width
    wheel_too_low = (wheel_x >= height_check_x) & (wheel_bottom_z < expected_surface_z - wheel_height_margin)
    if require_both_wheels_below:
        height_fall = torch.all(wheel_too_low, dim=1)
    else:
        height_fall = torch.any(wheel_too_low, dim=1)
    if not enable_height_fall_check:
        height_fall = torch.zeros_like(height_fall)

    return stage_gate & (~reached_top) & (lateral_fall | backed_off_start | height_fall)


def stage_base_contact(
    env: ManagerBasedRLEnv,
    stage: int,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor", body_names="base_link"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Return True when base_link contacts terrain on a selected terrain stage."""

    stage_gate = _stage_mask(env, stage)
    contact = torch.amax(contact_state_obs(env, sensor_cfg, threshold), dim=1) > 0.5
    return stage_gate & contact


def stage3_base_stair_contact(
    env: ManagerBasedRLEnv,
    stage: int = 3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    bottom_platform_width: float = 2.0,
    step_height: float = 0.10,
    step_width: float = 0.35,
    num_steps: int = 10,
    stair_width: float = 2.0,
    contact_margin: float = 0.01,
    base_half_length: float = 0.115,
    base_half_width: float = 0.09,
    base_bottom_z_offset: float = -0.03,
    lateral_margin: float = 0.05,
) -> torch.Tensor:
    """Terminate when the base box bottom contacts the stage3 stair surface.

    The global ``base_contact`` term relies on contact forces. This geometric
    check is specific to the generated straight stair and uses the base box's
    bottom face because ``base_link`` is centered in x/y but not in z.
    """

    stage_gate = _stage_mask(env, stage)
    asset: Articulation = env.scene[asset_cfg.name]

    local_offsets = torch.tensor(
        [
            [0.0, 0.0, base_bottom_z_offset],
            [base_half_length, 0.0, base_bottom_z_offset],
            [-base_half_length, 0.0, base_bottom_z_offset],
            [0.0, base_half_width, base_bottom_z_offset],
            [0.0, -base_half_width, base_bottom_z_offset],
            [base_half_length, base_half_width, base_bottom_z_offset],
            [base_half_length, -base_half_width, base_bottom_z_offset],
            [-base_half_length, base_half_width, base_bottom_z_offset],
            [-base_half_length, -base_half_width, base_bottom_z_offset],
        ],
        dtype=asset.data.root_pos_w.dtype,
        device=env.device,
    )
    num_samples = local_offsets.shape[0]
    root_pos_w = asset.data.root_pos_w
    root_quat_w = asset.data.root_quat_w
    sample_offsets_w = quat_apply(
        root_quat_w.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, 4),
        local_offsets.unsqueeze(0).expand(env.num_envs, -1, -1).reshape(-1, 3),
    ).view(env.num_envs, num_samples, 3)
    sample_pos = root_pos_w.unsqueeze(1) + sample_offsets_w - env.scene.terrain.env_origins.unsqueeze(1)

    surface_z = _stage3_actual_stair_surface_height(
        sample_pos[..., 0],
        bottom_platform_width,
        step_height,
        step_width,
        num_steps,
    )
    inside_width = torch.abs(sample_pos[..., 1]) <= 0.5 * stair_width + lateral_margin
    base_too_low = sample_pos[..., 2] <= surface_z + contact_margin
    return stage_gate & torch.any(inside_width & base_too_low, dim=1)


def stage3_split_stair_stance_timeout(
    env: ManagerBasedRLEnv,
    stage: int = 3,
    wheel_body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["L_wheel_link", "R_wheel_link"]),
    bottom_platform_width: float = 2.0,
    step_height: float = 0.10,
    step_width: float = 0.35,
    num_steps: int = 10,
    wheel_radius: float = 0.0625,
    stance_height_margin: float = 0.04,
    max_duration_s: float = 1.2,
) -> torch.Tensor:
    """Terminate if the wheels stand on different stair levels for too long."""

    stage_gate = _stage_mask(env, stage)
    wheel_steps, wheel_on_surface = _stage3_wheel_stance_steps(
        env,
        wheel_body_cfg,
        bottom_platform_width,
        step_height,
        step_width,
        num_steps,
        wheel_radius,
        stance_height_margin,
    )
    both_in_stance = torch.all(wheel_on_surface, dim=1)
    different_steps = torch.abs(wheel_steps[:, 0] - wheel_steps[:, 1]) >= 1.0
    at_least_one_on_stairs = torch.amax(wheel_steps, dim=1) > 0.0
    split_stance = stage_gate & both_in_stance & different_steps & at_least_one_on_stairs

    if not hasattr(env, "wheelleg_stage3_split_stance_counter"):
        env.wheelleg_stage3_split_stance_counter = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
        env.wheelleg_stage3_split_stance_counter_step = -1

    reset_mask = env.episode_length_buf <= 1
    env.wheelleg_stage3_split_stance_counter[reset_mask] = 0

    current_step = int(env.common_step_counter)
    if env.wheelleg_stage3_split_stance_counter_step != current_step:
        env.wheelleg_stage3_split_stance_counter = torch.where(
            split_stance,
            env.wheelleg_stage3_split_stance_counter + 1,
            torch.zeros_like(env.wheelleg_stage3_split_stance_counter),
        )
        env.wheelleg_stage3_split_stance_counter_step = current_step

    max_steps = max(1, int(max_duration_s / env.step_dt))
    return env.wheelleg_stage3_split_stance_counter >= max_steps


def stage3_heading_error_timeout(
    env: ManagerBasedRLEnv,
    stage: int = 3,
    target_heading: float = 0.0,
    max_heading_error: float = 1.0472,
    max_duration_s: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate if stage3 heading is far from uphill for too long."""

    stage_gate = _stage_mask(env, stage)
    asset: Articulation = env.scene[asset_cfg.name]
    heading_error = torch.abs(wrap_to_pi(target_heading - asset.data.heading_w))
    bad_heading = stage_gate & (heading_error > max_heading_error)

    if not hasattr(env, "wheelleg_stage3_bad_heading_counter"):
        env.wheelleg_stage3_bad_heading_counter = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
        env.wheelleg_stage3_bad_heading_counter_step = -1

    reset_mask = env.episode_length_buf <= 1
    env.wheelleg_stage3_bad_heading_counter[reset_mask] = 0

    current_step = int(env.common_step_counter)
    if env.wheelleg_stage3_bad_heading_counter_step != current_step:
        env.wheelleg_stage3_bad_heading_counter = torch.where(
            bad_heading,
            env.wheelleg_stage3_bad_heading_counter + 1,
            torch.zeros_like(env.wheelleg_stage3_bad_heading_counter),
        )
        env.wheelleg_stage3_bad_heading_counter_step = current_step

    max_steps = max(1, int(max_duration_s / env.step_dt))
    return env.wheelleg_stage3_bad_heading_counter >= max_steps
