# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Alternating-walk task configuration for the WheelLeg robot.

This task inherits the common robot scene, actions, reset events, termination
terms, and simulation settings from ``wheelleg_env_cfg.py``. It only overrides
the parts that define the alternating gait behavior:

* command range: start from low-speed forward rolling;
* observations: add base velocity, phase clock, contact state, and wheel speed;
* rewards: add tracking, swing, stance, wheel continuity, and smoothness terms.
"""

from __future__ import annotations

import math

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .mdp.observations import contact_state_obs, gait_phase_obs, wheel_speed_obs
from .mdp.rewards import (
    alternating_contact_reward,
    base_height_oscillation_reward,
    hip_action_sync_penalty,
    swing_clearance_reward,
    swing_drag_penalty,
    wheel_not_stop_penalty,
    wheel_rolling_consistency,
)
from .wheelleg_env_cfg import (
    CommandsCfg as BaseCommandsCfg,
    ObservationsCfg as BaseObservationsCfg,
    RewardsCfg as BaseRewardsCfg,
    WheellegEnvCfg,
)


# -----------------------------------------------------------------------------
# Task-specific config overrides
# -----------------------------------------------------------------------------


@configclass
class AlternatingWalkCommandsCfg(BaseCommandsCfg):
    """Low-speed forward rolling command range for the first gait curriculum."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 8.0),
        rel_standing_envs=0.01,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.2, 0.6),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.2, 0.2),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class AlternatingWalkObservationsCfg(BaseObservationsCfg):
    @configclass
    class PolicyCfg(BaseObservationsCfg.PolicyCfg):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        gait_phase = ObsTerm(
            func=gait_phase_obs,
            params={"gait_period": 0.6},
        )
        contact_state = ObsTerm(
            func=contact_state_obs,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_sensor",
                    body_names=["L_wheel_link", "R_wheel_link"],
                ),
                "threshold": 1.0,
            },
        )
        wheel_speed = ObsTerm(
            func=wheel_speed_obs,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["L_wheel_joint", "R_wheel_joint"],
                )
            },
        )

        def __post_init__(self) -> None:
            super().__post_init__()
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class AlternatingWalkRewardsCfg(BaseRewardsCfg):
    # Forward tracking and body stability are inherited from BaseRewardsCfg.
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.35},
    )

    wheel_rolling = RewTerm(
        func=wheel_rolling_consistency,
        weight=0.3,
        params={
            "command_name": "base_velocity",
            "wheel_radius": 0.1,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["L_wheel_joint", "R_wheel_joint"],
            ),
            "std": 0.35,
        },
    )
    wheel_stop = RewTerm(
        func=wheel_not_stop_penalty,
        weight=-0.3,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["L_wheel_joint", "R_wheel_joint"],
            ),
        },
    )
    base_height_bob = RewTerm(
        func=base_height_oscillation_reward,
        weight=0.6,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "gait_period": 0.6,
            "nominal_height": 0.5,
            "amplitude": 0.04,
            "height_std": 0.03,
            "velocity_std": 0.35,
            "upright_std": 0.25,
        },
    )
    stance_contact = RewTerm(
        func=alternating_contact_reward,
        weight=1.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=["L_wheel_link", "R_wheel_link"],
            ),
            "gait_period": 0.6,
            "threshold": 1.0,
            "double_support_width": 0.08,
        },
    )
    swing_clearance = RewTerm(
        func=swing_clearance_reward,
        weight=0.4,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["L_wheel_link", "R_wheel_link"],
            ),
            "gait_period": 0.6,
            "clearance_height": 0.18,
            "std": 0.05,
        },
    )
    swing_drag = RewTerm(
        func=swing_drag_penalty,
        weight=-0.8,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=["L_wheel_link", "R_wheel_link"],
            ),
            "gait_period": 0.6,
            "force_scale": 20.0,
        },
    )
    hip_sync = RewTerm(func=hip_action_sync_penalty, weight=-0.15)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0.8)
    base_too_low = RewTerm(func=mdp.root_height_below_minimum, weight=-1.0, params={"minimum_height": 0.01})
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=0.05)


@configclass
class WheellegAlternatingWalkEnvCfg(WheellegEnvCfg):
    """WheelLeg environment specialized for left/right alternating walking."""

    observations: AlternatingWalkObservationsCfg = AlternatingWalkObservationsCfg()
    commands: AlternatingWalkCommandsCfg = AlternatingWalkCommandsCfg()
    rewards: AlternatingWalkRewardsCfg = AlternatingWalkRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 5
