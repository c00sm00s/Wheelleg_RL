# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

from .observations import (
    contact_first_contact,
    contact_last_air_time,
    contact_sensor_forces,
    contact_state_obs,
    leg_phase_mask,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def wheel_rolling_consistency(
    env: ManagerBasedRLEnv,
    command_name: str,
    wheel_radius: float,
    asset_cfg: SceneEntityCfg,
    std: float = 0.35,
) -> torch.Tensor:
    """Reward wheel rolling speed that is consistent with commanded forward speed.

    If the USD uses opposite signs for the two wheel joints, change the forward
    speed estimate to ``0.5 * (left_wheel - right_wheel) * wheel_radius``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    wheel_forward_speed = torch.mean(wheel_vel, dim=-1) * wheel_radius
    target_vx = env.command_manager.get_command(command_name)[:, 0]
    error = torch.square(wheel_forward_speed - target_vx)
    return torch.exp(-error / (std * std))


def wheel_not_stop_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    min_command_speed: float = 0.15,
    min_wheel_speed: float = 0.5,
) -> torch.Tensor:
    """Penalize cases where a forward command exists but both wheels nearly stop."""
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_vel_abs = torch.mean(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=-1)
    target_vx_abs = torch.abs(env.command_manager.get_command(command_name)[:, 0])
    should_roll = target_vx_abs > min_command_speed
    is_stopped = wheel_vel_abs < min_wheel_speed
    return (should_roll & is_stopped).to(dtype=torch.float32)


def base_height_oscillation_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    gait_period: float = 0.6,
    nominal_height: float = 0.5,
    amplitude: float = 0.04,
    height_std: float = 0.03,
    velocity_std: float = 0.35,
    upright_std: float = 0.25,
) -> torch.Tensor:
    """Reward periodic base height motion while tracking velocity and staying upright."""
    asset: Articulation = env.scene[asset_cfg.name]

    time_s = env.episode_length_buf.to(dtype=torch.float32) * env.step_dt
    phase = 2.0 * math.pi * time_s / gait_period
    desired_height = nominal_height + amplitude * torch.sin(phase)

    height_error = asset.data.root_pos_w[:, 2] - desired_height
    height_reward = torch.exp(-torch.square(height_error) / (height_std * height_std))

    command = env.command_manager.get_command(command_name)
    velocity_error = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2] - command[:, :2]), dim=-1)
    velocity_gate = torch.exp(-velocity_error / (velocity_std * velocity_std))

    upright_error = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=-1)
    upright_gate = torch.exp(-upright_error / (upright_std * upright_std))

    return height_reward * velocity_gate * upright_gate


def alternating_contact_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    gait_period: float = 0.6,
    threshold: float = 1.0,
    double_support_width: float = 0.08,
) -> torch.Tensor:
    """Reward phase-matched left/right single support contact.

    The stance side should be in contact and the swing side should be off the
    ground. Double support is tolerated only near phase switches.
    """
    contact = contact_state_obs(env, sensor_cfg, threshold)
    left_contact = contact[:, 0]
    right_contact = contact[:, 1]

    left_phase, right_phase, left_swing, right_swing, left_stance, right_stance = leg_phase_mask(env, gait_period)
    phase_margin = torch.minimum(torch.abs(left_phase), torch.abs(right_phase))
    double_support = (phase_margin < double_support_width).to(dtype=torch.float32)

    stance_contact = left_stance * left_contact + right_stance * right_contact
    swing_clear = left_swing * (1.0 - left_contact) + right_swing * (1.0 - right_contact)
    swing_contact = left_swing * left_contact + right_swing * right_contact
    double_contact = left_contact * right_contact
    outside_transfer = 1.0 - double_support

    score = 0.5 * (stance_contact + swing_clear)
    score -= 0.5 * outside_transfer * swing_contact
    score -= 0.25 * outside_transfer * double_contact
    return torch.clamp(score, min=-1.0, max=1.0)


def swing_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    gait_period: float = 0.6,
    clearance_height: float = 0.18,
    std: float = 0.05,
) -> torch.Tensor:
    """Reward the swing-side foot/wheel body for clearing the ground.

    This assumes flat ground at z=0. For uneven terrain, replace world z with a
    terrain-relative foot height query.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    left_height = body_height[:, 0]
    right_height = body_height[:, 1]
    _, _, left_swing, right_swing, _, _ = leg_phase_mask(env, gait_period)

    left_error = torch.relu(clearance_height - left_height)
    right_error = torch.relu(clearance_height - right_height)
    left_reward = torch.exp(-torch.square(left_error) / (std * std))
    right_reward = torch.exp(-torch.square(right_error) / (std * std))
    swing_count = torch.clamp(left_swing + right_swing, min=1.0)
    return (left_swing * left_reward + right_swing * right_reward) / swing_count


def swing_drag_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
    gait_period: float = 0.6,
    force_scale: float = 20.0,
) -> torch.Tensor:
    """Penalize large contact force on the side that should be swinging."""
    force_norm = torch.linalg.norm(contact_sensor_forces(env, sensor_cfg), dim=-1)
    left_force = force_norm[:, 0]
    right_force = force_norm[:, 1]

    _, _, left_swing, right_swing, _, _ = leg_phase_mask(env, gait_period)
    swing_force = left_swing * left_force + right_swing * right_force
    return torch.tanh(swing_force / force_scale)


def hip_action_sync_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize left and right hip actions moving too synchronously.

    Base action order:
    [L_hip_front, L_hip_rear, R_hip_front, R_hip_rear, L_wheel, R_wheel]
    """
    action = env.action_manager.action
    left_hip = torch.mean(action[:, 0:2], dim=-1)
    right_hip = torch.mean(action[:, 2:4], dim=-1)
    return torch.relu(left_hip * right_hip)


def jump_hip_coordination_reward(
    env: ManagerBasedRLEnv,
    jump_command_name: str = "jump_command",
    same_side_std: float = 0.35,
    left_right_std: float = 0.35,
    anticipation_time: float = 0.45,
) -> torch.Tensor:
    """Reward five-bar hip coordination for compression and extension.

    Action layout for the jump task:
    [L_hip_front, L_hip_rear, R_hip_front, R_hip_rear, L_wheel, R_wheel].

    For each side, front/rear hip efforts should oppose each other to open or
    close the linkage. Since the left/right mechanisms are mirrored, their
    linkage efforts should be opposite in action coordinates for symmetric body
    motion.
    """
    command = env.command_manager.get_command(jump_command_name)
    mask = _jump_prepare(command, anticipation_time)

    action = env.action_manager.action
    left_front = action[:, 0]
    left_rear = action[:, 1]
    right_front = action[:, 2]
    right_rear = action[:, 3]

    left_mode = 0.5 * (left_front - left_rear)
    right_mode = 0.5 * (right_front - right_rear)
    left_waste = 0.5 * (left_front + left_rear)
    right_waste = 0.5 * (right_front + right_rear)

    same_side_error = torch.square(left_waste) + torch.square(right_waste)
    left_right_error = torch.square(left_mode + right_mode)
    same_side_reward = torch.exp(-same_side_error / (same_side_std * same_side_std))
    left_right_reward = torch.exp(-left_right_error / (left_right_std * left_right_std))

    return mask * same_side_reward * left_right_reward


def jump_wheel_sagittal_alignment_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    jump_command_name: str = "jump_command",
    x_std: float = 0.04,
    z_std: float = 0.03,
    anticipation_time: float = 0.45,
) -> torch.Tensor:
    """Reward left/right wheels staying aligned in sagittal position and leg length."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(jump_command_name)
    mask = _jump_prepare(command, anticipation_time)

    wheel_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    left_x = wheel_pos[:, 0, 0] - asset.data.root_pos_w[:, 0]
    right_x = wheel_pos[:, 1, 0] - asset.data.root_pos_w[:, 0]
    left_z = wheel_pos[:, 0, 2] - asset.data.root_pos_w[:, 2]
    right_z = wheel_pos[:, 1, 2] - asset.data.root_pos_w[:, 2]

    x_reward = torch.exp(-torch.square(left_x - right_x) / (x_std * x_std))
    z_reward = torch.exp(-torch.square(left_z - right_z) / (z_std * z_std))
    return mask * x_reward * z_reward


def jump_contact_symmetry_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    jump_command_name: str = "jump_command",
    threshold: float = 1.0,
    anticipation_time: float = 0.45,
    air_phase_start: float = 0.18,
    air_phase_width: float = 0.20,
) -> torch.Tensor:
    """Reward both wheels changing contact state together during a jump.

    The desired contact pattern is:
    - preparation / early push-off: both wheels still touch the ground;
    - flight: both wheels are off the ground.

    This prevents the common early-training failure where one side unloads
    first, twists the body, and the policy learns rolling instead of a vertical
    hop.
    """
    command = env.command_manager.get_command(jump_command_name)
    mask = _jump_prepare(command, anticipation_time)
    contact = contact_state_obs(env, sensor_cfg, threshold)

    left_contact = contact[:, 0]
    right_contact = contact[:, 1]
    both_ground = left_contact * right_contact
    both_air = (1.0 - left_contact) * (1.0 - right_contact)
    same_state = 1.0 - torch.abs(left_contact - right_contact)

    phase = command[:, 2] if command.shape[-1] > 2 else command[:, 1]
    air_weight = torch.clamp((phase - air_phase_start) / max(air_phase_width, 1.0e-6), 0.0, 1.0)
    desired_pattern = (1.0 - air_weight) * both_ground + air_weight * both_air

    return mask * (0.45 * same_state + 0.55 * desired_pattern)


def wheel_contact_stability_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    jump_command_name: str = "jump_command",
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward both wheels staying in contact outside the active jump window."""
    contact = contact_state_obs(env, sensor_cfg, threshold)
    both_wheels = torch.prod(contact, dim=-1)
    jump_command = env.command_manager.get_command(jump_command_name)
    active = jump_command[:, 1] > 0.5
    return both_wheels * (~active).to(dtype=torch.float32)


def _jump_active(command: torch.Tensor) -> torch.Tensor:
    """Return mask for the active jump command window."""
    return (command[:, 1] > 0.5).to(dtype=torch.float32)


def _jump_started(command: torch.Tensor) -> torch.Tensor:
    """Return mask after the jump trigger has been reached."""
    enabled = command[:, 0] > 0.0
    if command.shape[-1] > 3:
        triggered = command[:, 3] <= 1.0e-6
    else:
        triggered = command[:, 1] > 0.5
    return (enabled & triggered).to(dtype=torch.float32)


def _jump_prepare(command: torch.Tensor, anticipation_time: float) -> torch.Tensor:
    """Return mask for late preparation or active jump command."""
    active = command[:, 1] > 0.5
    if command.shape[-1] <= 3 or anticipation_time <= 0.0:
        return active.to(dtype=torch.float32)

    enabled = command[:, 0] > 0.0
    time_to_jump = command[:, 3]
    preparing = enabled & (time_to_jump > 0.0) & (time_to_jump <= anticipation_time)
    return (active | preparing).to(dtype=torch.float32)


def jump_crouch_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    jump_command_name: str = "jump_command",
    crouch_height: float = 0.38,
    nominal_height: float = 0.5,
    height_std: float = 0.04,
    upright_std: float = 0.25,
    threshold: float = 1.0,
    anticipation_time: float = 0.35,
    min_air_time: float = 0.08,
) -> torch.Tensor:
    """Reward lowering the base while both wheels stay grounded before takeoff."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(jump_command_name)
    mask = _jump_prepare(command, anticipation_time)

    base_height = asset.data.root_pos_w[:, 2]
    height_reward = torch.exp(-torch.square(base_height - crouch_height) / (height_std * height_std))
    crouch_progress = torch.clamp(
        (nominal_height - base_height) / max(nominal_height - crouch_height, 1.0e-6),
        0.0,
        1.0,
    )
    upright_error = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=-1)
    upright_gate = torch.exp(-upright_error / (upright_std * upright_std))
    contact = torch.prod(contact_state_obs(env, sensor_cfg, threshold), dim=-1)
    had_flight = (torch.mean(contact_last_air_time(env, sensor_cfg), dim=-1) > min_air_time).to(dtype=torch.float32)
    return mask * (1.0 - had_flight) * (0.5 * height_reward + 0.5 * crouch_progress) * upright_gate * contact


def jump_push_off_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    jump_command_name: str = "jump_command",
    crouch_height: float = 0.44,
    nominal_height: float = 0.5,
    takeoff_speed: float = 0.8,
    speed_std: float = 0.35,
    threshold: float = 1.0,
    min_air_time: float = 0.08,
) -> torch.Tensor:
    """Reward upward velocity once the body is compressed or wheels start unloading."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(jump_command_name)
    mask = _jump_active(command)

    base_height = asset.data.root_pos_w[:, 2]
    compression = torch.clamp((nominal_height - base_height) / max(nominal_height - crouch_height, 1.0e-6), 0.0, 1.0)
    up_speed = asset.data.root_lin_vel_w[:, 2]
    speed_reward = torch.exp(-torch.square(up_speed - takeoff_speed) / (speed_std * speed_std))
    speed_progress = torch.clamp(up_speed / takeoff_speed, 0.0, 1.0)
    contact = contact_state_obs(env, sensor_cfg, threshold)
    mean_contact = torch.mean(contact, dim=-1)
    airborne = 1.0 - torch.prod(contact, dim=-1)
    unload = 1.0 - mean_contact
    push_ready = torch.maximum(compression, unload)
    had_flight = (torch.mean(contact_last_air_time(env, sensor_cfg), dim=-1) > min_air_time).to(dtype=torch.float32)
    return (
        mask
        * (1.0 - had_flight)
        * push_ready
        * (0.5 * speed_reward + 0.5 * speed_progress)
        * (0.5 + 0.5 * airborne)
    )


def jump_wheel_liftoff_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    jump_command_name: str = "jump_command",
    wheel_radius: float = 0.0625,
    height_std: float = 0.05,
    upright_std: float = 0.35,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward the lower of the two wheel bottoms clearing the ground.

    The task definition treats a jump as successful only when the lowest wheel
    point leaves the ground. Using the minimum of left/right wheel-bottom
    heights makes the reward strict: one wheel high and the other dragging does
    not count as a jump.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(jump_command_name)
    mask = _jump_active(command)

    wheel_center_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    wheel_bottom_height = wheel_center_height - wheel_radius
    min_wheel_bottom_height = torch.min(wheel_bottom_height, dim=-1).values

    target_height = torch.clamp(command[:, 0], min=1.0e-6)
    height_error = min_wheel_bottom_height - target_height
    height_reward = torch.exp(-torch.square(height_error) / (height_std * height_std))
    height_progress = torch.clamp(min_wheel_bottom_height / target_height, 0.0, 1.0)

    contact = contact_state_obs(env, sensor_cfg, threshold)
    both_air = torch.prod(1.0 - contact, dim=-1)
    same_contact_state = 1.0 - torch.abs(contact[:, 0] - contact[:, 1])
    upright_error = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=-1)
    upright_gate = torch.exp(-upright_error / (upright_std * upright_std))

    # Give progress while the robot is learning to unload, but reserve the full
    # reward for simultaneous two-wheel flight.
    clearance_score = 0.45 * height_reward + 0.55 * height_progress
    flight_gate = 0.35 + 0.65 * both_air
    return mask * clearance_score * flight_gate * same_contact_state * upright_gate


def jump_flight_height_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    jump_command_name: str = "jump_command",
    wheel_radius: float = 0.0625,
    height_std: float = 0.05,
    upright_std: float = 0.25,
    takeoff_clearance: float = 0.02,
) -> torch.Tensor:
    """Reward both wheel bottoms reaching the commanded height after takeoff."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(jump_command_name)
    # Wheel link z-height minus wheel radius gives the physical wheel-bottom
    # height. The minimum over left/right is the jump success signal.
    wheel_center_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    wheel_bottom_height = wheel_center_height - wheel_radius
    min_wheel_bottom_height = torch.min(wheel_bottom_height, dim=-1).values
    mask = _jump_started(command)
    target_height = command[:, 0]
    clearance = (min_wheel_bottom_height > takeoff_clearance).to(dtype=torch.float32)
    height_error = min_wheel_bottom_height - target_height
    height_reward = torch.exp(-torch.square(height_error) / (height_std * height_std))
    height_progress = torch.clamp((min_wheel_bottom_height) / torch.clamp(command[:, 0], min=1.0e-6), 0.0, 1.0)
    upright_error = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=-1)
    upright_gate = torch.exp(-upright_error / (upright_std * upright_std))
    return mask * clearance * (0.5 * height_reward + 0.5 * height_progress) * upright_gate


def jump_wheel_landing_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    jump_command_name: str = "jump_command",
    threshold: float = 1.0,
    min_air_time: float = 0.08,
) -> torch.Tensor:
    """Reward first wheel landing contact after a real airborne interval."""
    command = env.command_manager.get_command(jump_command_name)
    mask = _jump_started(command)
    contact = contact_state_obs(env, sensor_cfg, threshold)
    both_wheels = torch.prod(contact, dim=-1)
    balanced_force = 1.0 - torch.abs(contact[:, 0] - contact[:, 1])
    first_contact = torch.clamp(torch.sum(contact_first_contact(env, sensor_cfg), dim=-1), max=1.0)
    air_time = contact_last_air_time(env, sensor_cfg)
    had_flight = (torch.mean(air_time, dim=-1) > min_air_time).to(dtype=torch.float32)
    return mask * first_contact * both_wheels * balanced_force * had_flight


def jump_recovery_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    jump_command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    nominal_height: float = 0.5,
    height_std: float = 0.05,
    velocity_std: float = 0.35,
    upright_std: float = 0.25,
    threshold: float = 1.0,
    min_air_time: float = 0.08,
) -> torch.Tensor:
    """Reward returning to normal rolling after a detected flight and landing."""
    asset: Articulation = env.scene[asset_cfg.name]
    jump_command = env.command_manager.get_command(jump_command_name)
    base_command = env.command_manager.get_command(command_name)
    contact = torch.prod(contact_state_obs(env, sensor_cfg, threshold), dim=-1)
    had_flight = (torch.mean(contact_last_air_time(env, sensor_cfg), dim=-1) > min_air_time).to(dtype=torch.float32)
    recovery = _jump_started(jump_command) * had_flight * contact

    height_reward = torch.exp(-torch.square(asset.data.root_pos_w[:, 2] - nominal_height) / (height_std * height_std))
    velocity_error = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2] - base_command[:, :2]), dim=-1)
    velocity_gate = torch.exp(-velocity_error / (velocity_std * velocity_std))
    upright_error = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=-1)
    upright_gate = torch.exp(-upright_error / (upright_std * upright_std))
    return recovery * height_reward * velocity_gate * upright_gate
