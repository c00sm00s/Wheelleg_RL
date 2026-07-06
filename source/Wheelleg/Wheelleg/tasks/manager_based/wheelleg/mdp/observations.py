# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _phase(env: ManagerBasedRLEnv, gait_period: float) -> torch.Tensor:
    """Return phase clock angle phi = 2*pi*t/T,
    where T is the gait period
    and t is the time since the episode start."""
    time_s = env.episode_length_buf.to(dtype=torch.float32) * env.step_dt
    return torch.remainder(2.0 * math.pi * time_s / gait_period, 2 * math.pi)


def leg_phase_mask(
        env: ManagerBasedRLEnv, gait_period: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return phase clock values and left/right swing/stance masks.

    P_L = sin(phi), P_R = sin(phi + pi).
    P > 0 means swing, and P <= 0 means stance.
    """
    phi = _phase(env, gait_period)
    left_phase = torch.sin(phi)
    right_phase = torch.sin(phi + math.pi)
    left_swing = (left_phase > 0).to(dtype=torch.float32)
    right_swing = (right_phase > 0).to(dtype=torch.float32)
    left_stance = 1.0 - left_swing
    right_stance = 1.0 - right_swing
    return left_phase, right_phase, left_swing, right_swing, left_stance, right_stance


def asymmetric_gait_phase_mask(
    env: ManagerBasedRLEnv,
    gait_period: float,
    swing_fraction: float = 0.35,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return an alternating clock with short swing and long stance phases.

    Left swing occupies the first part of the cycle, right swing starts halfway
    through the cycle, and the gaps are double-support windows.
    """

    time_s = env.episode_length_buf.to(dtype=torch.float32) * env.step_dt
    cycle = torch.remainder(time_s / gait_period, 1.0)
    swing_fraction = min(max(swing_fraction, 1.0e-3), 0.49)

    left_swing = (cycle < swing_fraction).to(dtype=torch.float32)
    right_swing = ((cycle >= 0.5) & (cycle < 0.5 + swing_fraction)).to(dtype=torch.float32)
    left_stance = 1.0 - left_swing
    right_stance = 1.0 - right_swing
    phase = 2.0 * math.pi * cycle
    return phase, torch.sin(phase), torch.cos(phase), left_swing, right_swing, left_stance, right_stance


def blocked_asymmetric_gait_phase_mask(
    env: ManagerBasedRLEnv,
    blocked_gate: torch.Tensor,
    gait_period: float,
    swing_fraction: float = 0.35,
    state_prefix: str = "wheelleg_blocked_asymmetric_gait",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return an asymmetric clock that starts only after a side is blocked.

    The blocked side enters swing immediately while the other side remains in
    stance. When no side is blocked, the clock resets and all outputs are zero.
    """

    swing_fraction = min(max(swing_fraction, 1.0e-3), 0.49)
    dt_ratio = env.step_dt / max(gait_period, 1.0e-6)
    blocked_left = blocked_gate[:, 0] > 0.0
    blocked_right = (blocked_gate[:, 1] > 0.0) & ~blocked_left
    blocked_active = blocked_left | blocked_right
    blocked_side = torch.full((env.num_envs,), -1, dtype=torch.int32, device=env.device)
    blocked_side[blocked_left] = 0
    blocked_side[blocked_right] = 1
    start_cycle = torch.zeros(env.num_envs, device=env.device)

    phase_attr = f"{state_prefix}_cycle"
    active_attr = f"{state_prefix}_active"
    side_attr = f"{state_prefix}_side"
    step_attr = f"{state_prefix}_step"
    if not hasattr(env, phase_attr):
        setattr(env, phase_attr, torch.zeros(env.num_envs, dtype=torch.float32, device=env.device))
        setattr(env, active_attr, torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
        setattr(env, side_attr, torch.full((env.num_envs,), -1, dtype=torch.int32, device=env.device))
        setattr(env, step_attr, -1)

    cycle = getattr(env, phase_attr)
    active = getattr(env, active_attr)
    side = getattr(env, side_attr)
    current_step = int(env.common_step_counter)
    if getattr(env, step_attr) != current_step:
        reset_mask = env.episode_length_buf <= 1
        side_changed = blocked_active & (side != blocked_side)
        newly_active = blocked_active & ((~active) | side_changed)
        continuing = blocked_active & (~newly_active)

        cycle = torch.where(continuing, torch.remainder(cycle + dt_ratio, 1.0), cycle)
        cycle = torch.where(newly_active, start_cycle, cycle)
        cycle = torch.where(blocked_active & (~reset_mask), cycle, torch.zeros_like(cycle))
        active = blocked_active & (~reset_mask)
        side = torch.where(active, blocked_side, torch.full_like(side, -1))

        setattr(env, phase_attr, cycle)
        setattr(env, active_attr, active)
        setattr(env, side_attr, side)
        setattr(env, step_attr, current_step)

    phase = 2.0 * math.pi * cycle
    left_swing = ((cycle < swing_fraction) & active & (side == 0)).to(dtype=torch.float32)
    right_swing = ((cycle < swing_fraction) & active & (side == 1)).to(dtype=torch.float32)
    left_stance = ((active) & (left_swing <= 0.0)).to(dtype=torch.float32)
    right_stance = ((active) & (right_swing <= 0.0)).to(dtype=torch.float32)
    return phase, torch.sin(phase) * active.to(dtype=torch.float32), torch.cos(phase) * active.to(dtype=torch.float32), left_swing, right_swing, left_stance, right_stance


def gait_phase_obs(env: ManagerBasedRLEnv, gait_period: float) -> torch.Tensor:
    """Return phase clock signals and swing/stance flags.

    Observation layout:
    [sin(phi), cos(phi), P_L, P_R, L_swing, R_swing, L_stance, R_stance].
    """
    phi = _phase(env, gait_period)
    left_phase, right_phase, left_swing, right_swing, left_stance, right_stance = leg_phase_mask(env, gait_period)
    return torch.stack(
        (
            torch.sin(phi),
            torch.cos(phi),
            left_phase,
            right_phase,
            left_swing,
            right_swing,
            left_stance,
            right_stance,
        ),
        dim=-1,
    )


def stage_asymmetric_gait_phase_obs(
    env: ManagerBasedRLEnv,
    stage: int | None = None,
    gait_period: float = 0.8,
    swing_fraction: float = 0.35,
    blocked_latch_attr: str | None = None,
    state_prefix: str = "wheelleg_blocked_asymmetric_gait",
) -> torch.Tensor:
    """Return a stage-gated asymmetric gait clock observation.

    Observation layout:
    [sin(phase), cos(phase), L_swing, R_swing, L_stance, R_stance].
    """

    if blocked_latch_attr is None:
        _, phase_sin, phase_cos, left_swing, right_swing, left_stance, right_stance = asymmetric_gait_phase_mask(
            env,
            gait_period,
            swing_fraction,
        )
    else:
        blocked_gate = getattr(env, blocked_latch_attr, None)
        if blocked_gate is None:
            blocked_gate = torch.zeros(env.num_envs, 2, dtype=torch.float32, device=env.device)
        _, phase_sin, phase_cos, left_swing, right_swing, left_stance, right_stance = blocked_asymmetric_gait_phase_mask(
            env,
            blocked_gate.to(dtype=torch.float32),
            gait_period,
            swing_fraction,
            state_prefix,
        )
    obs = torch.stack((phase_sin, phase_cos, left_swing, right_swing, left_stance, right_stance), dim=-1)

    if stage is None:
        return obs

    terrain = getattr(env.scene, "terrain", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    if terrain_types is None:
        return torch.zeros_like(obs)
    stage_gate = (terrain_types == stage).to(dtype=torch.float32).unsqueeze(1)
    return obs * stage_gate


def _contact_body_ids(sensor: ContactSensor, sensor_cfg: SceneEntityCfg) -> list[int] | slice:
    """Resolve contact sensor body ids from body_names when Isaac Lab leaves body_ids as all bodies."""
    if sensor_cfg.body_names is None or not isinstance(sensor_cfg.body_ids, slice):
        return sensor_cfg.body_ids

    if sensor_cfg.body_ids != slice(None):
        return sensor_cfg.body_ids

    requested_names = sensor_cfg.body_names
    if isinstance(requested_names, str):
        requested_names = [requested_names]

    sensor_body_names = getattr(sensor, "body_names", None) or getattr(sensor, "_body_names", None)
    if sensor_body_names is None:
        raise RuntimeError(
            f"Contact sensor '{sensor_cfg.name}' does not expose body_names, "
            f"so requested bodies {requested_names} cannot be selected."
        )

    body_ids = []
    for requested_name in requested_names:
        matches = [
            body_id
            for body_id, sensor_body_name in enumerate(sensor_body_names)
            if sensor_body_name == requested_name or sensor_body_name.rsplit("/", 1)[-1] == requested_name
        ]
        if not matches:
            raise RuntimeError(
                f"Body '{requested_name}' was not found in contact sensor '{sensor_cfg.name}'. "
                f"Available bodies: {sensor_body_names}."
            )
        body_ids.append(matches[0])

    return body_ids


def contact_sensor_forces(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return contact forces for the selected contact sensor bodies."""
    sensor = cast(ContactSensor, env.scene.sensors[sensor_cfg.name])
    body_ids = _contact_body_ids(sensor, sensor_cfg)

    if sensor.data.net_forces_w is None:
        if isinstance(body_ids, slice):
            num_bodies = 0
        else:
            num_bodies = len(body_ids)
        return torch.zeros(env.num_envs, num_bodies, 3, dtype=torch.float32, device=env.device)

    return sensor.data.net_forces_w[:, body_ids, :]


def contact_state_obs(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Return contact state observation for left and right legs.

    Observation layout:
    [L_contact, R_contact].
    """
    forces = contact_sensor_forces(env, sensor_cfg)
    contact = torch.linalg.norm(forces, dim=-1) > threshold
    return contact.to(dtype=torch.float32)


def leg_link_contact_state_obs(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    links_per_side: int = 4,
) -> torch.Tensor:
    """Return left/right non-wheel leg-link contact state.

    The configured body list is expected to contain all left-side leg links
    first, then all right-side leg links. Observation layout:
    [L_link_contact, R_link_contact].
    """

    contact = contact_state_obs(env, sensor_cfg, threshold)
    if contact.shape[1] < 2 * links_per_side:
        return torch.zeros(env.num_envs, 2, dtype=torch.float32, device=env.device)

    left_contact = torch.amax(contact[:, :links_per_side], dim=1)
    right_contact = torch.amax(contact[:, links_per_side : 2 * links_per_side], dim=1)
    return torch.stack((left_contact, right_contact), dim=1)


def contact_last_air_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return last air time for selected contact sensor bodies."""
    sensor = cast(ContactSensor, env.scene.sensors[sensor_cfg.name])
    body_ids = _contact_body_ids(sensor, sensor_cfg)

    if sensor.data.last_air_time is None:
        if isinstance(body_ids, slice):
            num_bodies = len(getattr(sensor, "body_names", []) or [])
        else:
            num_bodies = len(body_ids)
        return torch.zeros(env.num_envs, num_bodies, dtype=torch.float32, device=env.device)

    return sensor.data.last_air_time[:, body_ids]


def contact_first_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, dt: float | None = None) -> torch.Tensor:
    """Return first-contact events for selected contact sensor bodies."""
    sensor = cast(ContactSensor, env.scene.sensors[sensor_cfg.name])
    body_ids = _contact_body_ids(sensor, sensor_cfg)
    first_contact = sensor.compute_first_contact(env.step_dt if dt is None else dt)
    return first_contact[:, body_ids].to(dtype=torch.float32)


def wheel_speed_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return wheel speed observation for left and right wheels.

    Observation layout:
    [L_wheel_speed, R_wheel_speed].
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids]


def drive_motor_state_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return absolute drive motor angles and angular velocities.

    Observation layout for the configured drive joints:
    [q_0..q_n, qd_0..qd_n]. The angles are wrapped to [-pi, pi] so the
    policy sees a continuous revolute-joint state instead of unbounded turns.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.cat((joint_pos, joint_vel), dim=-1)


def wheel_end_effector_state_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return wheel/end-effector planar state in the base frame.

    The wheel links are the five-bar end effectors in this model. Using their
    simulated pose gives the policy direct local coordinates after the linkage
    kinematics are resolved by Isaac Sim. Observation layout per wheel is
    [x_b, z_b, vx_b, vz_b], flattened left then right.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    body_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    num_bodies = body_pos_w.shape[1]

    root_pos_w = asset.data.root_pos_w.unsqueeze(1)
    root_vel_w = asset.data.root_lin_vel_w.unsqueeze(1)
    root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, num_bodies, -1).reshape(-1, 4)

    rel_pos_w = (body_pos_w - root_pos_w).reshape(-1, 3)
    rel_vel_w = (body_vel_w - root_vel_w).reshape(-1, 3)
    rel_pos_b = quat_apply_inverse(root_quat_w, rel_pos_w).view(env.num_envs, num_bodies, 3)
    rel_vel_b = quat_apply_inverse(root_quat_w, rel_vel_w).view(env.num_envs, num_bodies, 3)

    planar_state = torch.stack((rel_pos_b[..., 0], rel_pos_b[..., 2], rel_vel_b[..., 0], rel_vel_b[..., 2]), dim=-1)
    return planar_state.reshape(env.num_envs, -1)


def foot_height_scan(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Return a foot-local terrain height scan from a ray-caster sensor.

    The output is the scanner height minus each terrain hit height, with a
    constant offset subtracted to keep values centered around flat terrain.
    """

    sensor = env.scene.sensors[sensor_cfg.name]
    heights = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    return torch.nan_to_num(heights, nan=0.0, posinf=0.0, neginf=0.0)


def action_history_obs(env: ManagerBasedRLEnv, history_length: int = 5) -> torch.Tensor:
    """Return a short history of recently applied actions.

    The newest action is stored first. The history lets the policy detect cases
    where wheel/hip commands changed but the end-effector state did not, which
    is a useful cue that a wheel is blocked by a stair and should be lifted.
    """

    action = env.action_manager.action
    action_dim = action.shape[1]
    needs_init = not hasattr(env, "wheelleg_action_history")
    if not needs_init:
        needs_init = env.wheelleg_action_history.shape[1:] != (history_length, action_dim)
    if needs_init:
        env.wheelleg_action_history = torch.zeros(env.num_envs, history_length, action_dim, device=env.device)
        env.wheelleg_action_history_step = -1

    reset_mask = env.episode_length_buf <= 1
    env.wheelleg_action_history[reset_mask] = 0.0

    current_step = int(env.common_step_counter)
    if env.wheelleg_action_history_step != current_step:
        env.wheelleg_action_history[:, 1:] = env.wheelleg_action_history[:, :-1].clone()
        env.wheelleg_action_history[:, 0] = action
        env.wheelleg_action_history_step = current_step

    return env.wheelleg_action_history.reshape(env.num_envs, -1)


def velocity_tracking_error_history_obs(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    history_length: int = 5,
) -> torch.Tensor:
    """Return recent body-frame XY velocity tracking errors.

    Observation layout per frame:
    [cmd_vx - base_vx, cmd_vy - base_vy, ||cmd_xy - base_xy||].
    The newest frame is stored first, so a policy can compare the current error
    against the previous few control steps and infer sudden blocking.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    command_xy = env.command_manager.get_command(command_name)[:, :2]
    error_xy = command_xy - asset.data.root_lin_vel_b[:, :2]
    error_norm = torch.linalg.norm(error_xy, dim=1, keepdim=True)
    error_frame = torch.cat((error_xy, error_norm), dim=1)
    frame_dim = error_frame.shape[1]

    needs_init = not hasattr(env, "wheelleg_velocity_error_history")
    if not needs_init:
        needs_init = env.wheelleg_velocity_error_history.shape[1:] != (history_length, frame_dim)
    if needs_init:
        env.wheelleg_velocity_error_history = error_frame.unsqueeze(1).repeat(1, history_length, 1)
        env.wheelleg_velocity_error_history_step = -1

    reset_mask = env.episode_length_buf <= 1
    env.wheelleg_velocity_error_history[reset_mask] = error_frame[reset_mask].unsqueeze(1)

    current_step = int(env.common_step_counter)
    if env.wheelleg_velocity_error_history_step != current_step:
        env.wheelleg_velocity_error_history[:, 1:] = env.wheelleg_velocity_error_history[:, :-1].clone()
        env.wheelleg_velocity_error_history[:, 0] = error_frame
        env.wheelleg_velocity_error_history_step = current_step

    return env.wheelleg_velocity_error_history.reshape(env.num_envs, -1)


def base_height_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return base/root height as a single observation."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2].unsqueeze(-1)
