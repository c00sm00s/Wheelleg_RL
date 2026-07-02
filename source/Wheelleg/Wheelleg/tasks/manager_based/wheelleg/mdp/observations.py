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


def base_height_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return base/root height as a single observation."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2].unsqueeze(-1)
