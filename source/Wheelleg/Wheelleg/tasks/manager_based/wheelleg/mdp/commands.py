# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific command generators for the WheelLeg robot."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    """Velocity command with a final range used by curriculum learning.

    Isaac Lab's built-in ``UniformVelocityCommandCfg`` only stores the current
    sampling range. The terrain task starts with small velocity commands and
    gradually expands them toward ``limit_ranges`` once tracking reward is good.
    """

    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
    """Maximum command range reached after the velocity curriculum converges."""


class StageAwareLevelVelocityCommand(UniformVelocityCommand):
    """Velocity command with a world-frame uphill override on the final stair stage."""

    cfg: "StageAwareLevelVelocityCommandCfg"

    def __init__(self, cfg: "StageAwareLevelVelocityCommandCfg", env):
        super().__init__(cfg, env)
        self.stage3_speed = torch.zeros(self.num_envs, device=self.device)

    def _stage3_env_ids(self) -> torch.Tensor:
        terrain = getattr(self._env.scene, "terrain", None)
        terrain_types = getattr(terrain, "terrain_types", None)
        if terrain_types is None:
            return torch.empty(0, dtype=torch.long, device=self.device)
        return (terrain_types == self.cfg.stage3_index).nonzero(as_tuple=False).flatten()

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        if len(env_ids) == 0:
            return

        if isinstance(env_ids, torch.Tensor):
            ids = env_ids.to(device=self.device, dtype=torch.long)
        else:
            ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        terrain = getattr(self._env.scene, "terrain", None)
        terrain_types = getattr(terrain, "terrain_types", None)
        if terrain_types is None:
            return

        stage3_mask = terrain_types[ids] == self.cfg.stage3_index
        if torch.any(stage3_mask):
            stage3_ids = ids[stage3_mask]
            random = torch.empty(len(stage3_ids), device=self.device)
            self.stage3_speed[stage3_ids] = random.uniform_(*self.cfg.stage3_lin_vel_x)
            self.is_standing_env[stage3_ids] = False

    def _update_command(self):
        super()._update_command()

        stage3_ids = self._stage3_env_ids()
        if len(stage3_ids) == 0:
            return

        world_dir = torch.tensor(self.cfg.stage3_world_direction, dtype=torch.float32, device=self.device)
        world_dir = world_dir / torch.clamp(torch.linalg.norm(world_dir), min=1.0e-6)
        vel_command_w = torch.zeros(len(stage3_ids), 3, dtype=torch.float32, device=self.device)
        vel_command_w[:, 0] = world_dir[0] * self.stage3_speed[stage3_ids]
        vel_command_w[:, 1] = world_dir[1] * self.stage3_speed[stage3_ids]

        yaw_quat = math_utils.yaw_quat(self.robot.data.root_quat_w[stage3_ids])
        vel_command_b = math_utils.quat_apply_inverse(yaw_quat, vel_command_w)
        self.vel_command_b[stage3_ids, 0] = vel_command_b[:, 0]
        self.vel_command_b[stage3_ids, 1] = vel_command_b[:, 1]

        target_heading = torch.atan2(world_dir[1], world_dir[0])
        heading_error = math_utils.wrap_to_pi(target_heading - self.robot.data.heading_w[stage3_ids])
        yaw_cmd = self.cfg.stage3_heading_control_stiffness * heading_error
        self.vel_command_b[stage3_ids, 2] = torch.clamp(
            yaw_cmd,
            min=self.cfg.stage3_ang_vel_z[0],
            max=self.cfg.stage3_ang_vel_z[1],
        )


@configclass
class StageAwareLevelVelocityCommandCfg(UniformLevelVelocityCommandCfg):
    """Stage-aware velocity command.

    Stage0-2 use normal random body-frame commands. Stage3 samples a positive
    world-frame uphill speed and converts it to body-frame command every step,
    so yaw drift cannot turn the command into a down-stair command.
    """

    class_type: type = StageAwareLevelVelocityCommand

    stage3_index: int = 3
    stage3_world_direction: tuple[float, float] = (1.0, 0.0)
    stage3_lin_vel_x: tuple[float, float] = (0.8, 1.5)
    stage3_heading_control_stiffness: float = 1.5
    stage3_ang_vel_z: tuple[float, float] = (-0.8, 0.8)


class TimedJumpCommand(CommandTerm):
    """Generate one timed jump-height command per episode.

    Command layout:
    ``[jump_height, jump_active, jump_phase, time_to_jump]``.
    """

    cfg: "TimedJumpCommandCfg"

    def __init__(self, cfg: "TimedJumpCommandCfg", env):
        super().__init__(cfg, env)
        self.jump_command = torch.zeros(self.num_envs, 4, device=self.device)
        self.jump_height = torch.zeros(self.num_envs, device=self.device)
        self.trigger_time = torch.zeros(self.num_envs, device=self.device)
        self.enabled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.metrics["jump_enabled"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["jump_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["jump_trigger_time"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.jump_command

    def _update_metrics(self):
        self.metrics["jump_enabled"][:] = self.enabled.to(dtype=torch.float32)
        self.metrics["jump_height"][:] = self.jump_height
        self.metrics["jump_trigger_time"][:] = self.trigger_time

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        random = torch.empty(len(env_ids), device=self.device)
        self.trigger_time[env_ids] = random.uniform_(*self.cfg.trigger_time_range)
        self.enabled[env_ids] = random.uniform_(0.0, 1.0) <= self.cfg.trigger_probability

        if self.cfg.jump_height_range is None:
            self.jump_height[env_ids] = self.cfg.jump_height
        else:
            self.jump_height[env_ids] = random.uniform_(*self.cfg.jump_height_range)

        self.jump_command[env_ids, 0] = self.jump_height[env_ids]
        self.jump_command[env_ids, 1:] = 0.0

    def _update_command(self):
        time_s = self._env.episode_length_buf.to(dtype=torch.float32) * self._env.step_dt
        time_from_trigger = time_s - self.trigger_time
        phase = torch.clamp(time_from_trigger / self.cfg.jump_duration, min=0.0, max=1.0)
        active = self.enabled & (time_from_trigger >= 0.0) & (time_from_trigger <= self.cfg.jump_duration)
        time_to_jump = torch.clamp(self.trigger_time - time_s, min=0.0)

        self.jump_command[:, 0] = torch.where(self.enabled, self.jump_height, 0.0)
        self.jump_command[:, 1] = active.to(dtype=torch.float32)
        self.jump_command[:, 2] = torch.where(self.enabled, phase, torch.zeros_like(phase))
        self.jump_command[:, 3] = torch.where(self.enabled, time_to_jump, torch.zeros_like(time_to_jump))

    def _set_debug_vis_impl(self, debug_vis: bool):
        raise NotImplementedError


@configclass
class TimedJumpCommandCfg(CommandTermCfg):
    """Configuration for :class:`TimedJumpCommand`."""

    class_type: type = TimedJumpCommand
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    trigger_time_range: tuple[float, float] = (5.0, 8.0)
    jump_height: float = 0.20
    # If set, sample the target wheel-bottom clearance once at reset.
    # Otherwise ``jump_height`` is used as a fixed target.
    jump_height_range: tuple[float, float] | None = None
    jump_duration: float = 1.8
    trigger_probability: float = 0.0
