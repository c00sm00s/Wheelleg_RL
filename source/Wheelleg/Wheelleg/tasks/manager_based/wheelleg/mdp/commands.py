# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific command generators for the WheelLeg robot."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.envs.mdp import UniformVelocityCommandCfg
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
