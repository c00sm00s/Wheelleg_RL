# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Timed jump task configuration for the WheelLeg robot."""

from __future__ import annotations

import math

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .mdp.observations import base_height_obs, contact_state_obs, wheel_speed_obs
from .mdp.rewards import (
    jump_contact_symmetry_reward,
    jump_crouch_reward,
    jump_flight_height_reward,
    jump_hip_coordination_reward,
    jump_push_off_reward,
    jump_recovery_reward,
    jump_wheel_liftoff_reward,
    jump_wheel_landing_reward,
    jump_wheel_sagittal_alignment_reward,
    wheel_contact_stability_reward,
    wheel_not_stop_penalty,
    wheel_rolling_consistency,
)
from .wheelleg_env_cfg import (
    ActionsCfg as BaseActionsCfg,
    CommandsCfg as BaseCommandsCfg,
    ObservationsCfg as BaseObservationsCfg,
    RewardsCfg as BaseRewardsCfg,
    WheellegEnvCfg,
)


WHEEL_CONTACT_CFG = SceneEntityCfg(
    "contact_sensor",
    body_names=["L_wheel_link", "R_wheel_link"],
)
ROBOT_CFG = SceneEntityCfg("robot")
WHEEL_JOINT_CFG = SceneEntityCfg(
    "robot",
    joint_names=["L_wheel_joint", "R_wheel_joint"],
)
WHEEL_BODY_CFG = SceneEntityCfg(
    "robot",
    body_names=["L_wheel_link", "R_wheel_link"],
)


@configclass
class JumpActionsCfg(BaseActionsCfg):
    """Expose active hip and wheel position targets for jump control.

    The knee joints belong to the leg linkage and are left passive/damped in
    the actuator setup. Driving them independently makes the front/rear links
    scissor through each other instead of compressing the leg.
    """

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "L_hip_front_joint",
            "L_hip_rear_joint",
            "R_hip_front_joint",
            "R_hip_rear_joint",
            "L_wheel_joint",
            "R_wheel_joint",
        ],
        scale={".*hip.*": 0.82905, ".*wheel.*": math.pi},
        offset={".*hip.*": -0.21815, ".*wheel.*": 0.0},
        clip={".*hip.*": (-1.0472, 0.6109), ".*wheel.*": (-math.pi, math.pi)},
        preserve_order=True,
        use_default_offset=False,
    )


@configclass
class JumpCommandsCfg(BaseCommandsCfg):
    """Low-speed rolling plus one randomized timed jump command."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 8.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            # Keep approach speed low. The task is vertical wheel-bottom
            # clearance, not fast rolling locomotion.
            lin_vel_x=(0.05, 0.20),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(-math.pi, math.pi),
        ),
    )
    jump_command = mdp.TimedJumpCommandCfg(
        # Trigger early enough that one episode contains preparation, takeoff,
        # landing, and recovery instead of mostly pre-jump rolling.
        trigger_time_range=(1.2, 2.0),
        jump_height=0.10,
        # Random target for the minimum wheel-bottom height above the ground.
        # Start modestly; after consistent liftoff, widen this range upward.
        jump_height_range=(0.06, 0.14),
        jump_duration=1.25,
        trigger_probability=1.0,
    )


@configclass
class JumpObservationsCfg(BaseObservationsCfg):
    """Policy observations for walking, jump timing, contact, and recovery."""

    @configclass
    class PolicyCfg(BaseObservationsCfg.PolicyCfg):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_height = ObsTerm(func=base_height_obs, params={"asset_cfg": ROBOT_CFG})
        jump_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "jump_command"})
        wheel_contact = ObsTerm(func=contact_state_obs, params={"sensor_cfg": WHEEL_CONTACT_CFG, "threshold": 1.0})
        wheel_speed = ObsTerm(func=wheel_speed_obs, params={"asset_cfg": WHEEL_JOINT_CFG})

        def __post_init__(self) -> None:
            super().__post_init__()
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class JumpRewardsCfg(BaseRewardsCfg):
    """Reward terms for a two-wheel synchronized vertical jump.

    The shaping is intentionally staged:
    1. before trigger: keep both wheels stable on the ground;
    2. preparation: compress both five-bar legs together;
    3. push-off: reward upward velocity and unloading;
    4. flight: reward the lower wheel bottom clearing the commanded height;
    5. landing: reward two-wheel contact and recovery to low-speed rolling.
    """

    alive = RewTerm(func=mdp.is_alive, weight=0.5)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=0.25,
        params={"command_name": "base_velocity", "std": 0.35},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.05,
        params={"command_name": "base_velocity", "std": 0.35},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    # Do not penalize vertical velocity in the jump task; upward speed is the
    # takeoff behavior we want. Upright and landing terms still keep it sane.
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.006)

    wheel_rolling = RewTerm(
        func=wheel_rolling_consistency,
        weight=0.03,
        params={
            "command_name": "base_velocity",
            "wheel_radius": 0.1,
            "asset_cfg": WHEEL_JOINT_CFG,
            "std": 0.35,
        },
    )
    wheel_stop = RewTerm(
        func=wheel_not_stop_penalty,
        weight=-0.01,
        params={"command_name": "base_velocity", "asset_cfg": WHEEL_JOINT_CFG},
    )
    wheel_contact_stability = RewTerm(
        func=wheel_contact_stability_reward,
        weight=0.6,
        params={"sensor_cfg": WHEEL_CONTACT_CFG, "jump_command_name": "jump_command", "threshold": 1.0},
    )
    jump_contact_symmetry = RewTerm(
        func=jump_contact_symmetry_reward,
        weight=2.0,
        params={
            "sensor_cfg": WHEEL_CONTACT_CFG,
            "jump_command_name": "jump_command",
            "threshold": 1.0,
            "anticipation_time": 0.55,
            "air_phase_start": 0.18,
            "air_phase_width": 0.20,
        },
    )
    jump_hip_coordination = RewTerm(
        func=jump_hip_coordination_reward,
        weight=1.2,
        params={
            "jump_command_name": "jump_command",
            "same_side_std": 0.35,
            "left_right_std": 0.35,
            "anticipation_time": 0.55,
        },
    )
    jump_wheel_sagittal_alignment = RewTerm(
        func=jump_wheel_sagittal_alignment_reward,
        weight=1.5,
        params={
            "asset_cfg": WHEEL_BODY_CFG,
            "jump_command_name": "jump_command",
            "x_std": 0.04,
            "z_std": 0.03,
            "anticipation_time": 0.55,
        },
    )

    jump_crouch = RewTerm(
        func=jump_crouch_reward,
        weight=4.5,
        params={
            "asset_cfg": ROBOT_CFG,
            "sensor_cfg": WHEEL_CONTACT_CFG,
            "jump_command_name": "jump_command",
            "crouch_height": 0.40,
            "nominal_height": 0.5,
            "height_std": 0.06,
            "anticipation_time": 0.55,
        },
    )
    jump_push_off = RewTerm(
        func=jump_push_off_reward,
        weight=8.0,
        params={
            "asset_cfg": ROBOT_CFG,
            "sensor_cfg": WHEEL_CONTACT_CFG,
            "jump_command_name": "jump_command",
            "crouch_height": 0.40,
            "nominal_height": 0.5,
            "takeoff_speed": 0.75,
            "speed_std": 0.35,
        },
    )
    jump_wheel_liftoff = RewTerm(
        func=jump_wheel_liftoff_reward,
        weight=14.0,
        params={
            "asset_cfg": WHEEL_BODY_CFG,
            "sensor_cfg": WHEEL_CONTACT_CFG,
            "jump_command_name": "jump_command",
            "wheel_radius": 0.0625,
            "height_std": 0.05,
            "threshold": 1.0,
        },
    )
    jump_flight_height = RewTerm(
        func=jump_flight_height_reward,
        weight=6.0,
        params={
            "asset_cfg": WHEEL_BODY_CFG,
            "jump_command_name": "jump_command",
            "wheel_radius": 0.0625,
            "height_std": 0.05,
            "takeoff_clearance": 0.005,
        },
    )
    jump_wheel_landing = RewTerm(
        func=jump_wheel_landing_reward,
        weight=2.0,
        params={"sensor_cfg": WHEEL_CONTACT_CFG, "jump_command_name": "jump_command", "threshold": 1.0},
    )
    jump_recovery = RewTerm(
        func=jump_recovery_reward,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "jump_command_name": "jump_command",
            "asset_cfg": ROBOT_CFG,
            "sensor_cfg": WHEEL_CONTACT_CFG,
            "nominal_height": 0.5,
        },
    )


@configclass
class WheellegJumpEnvCfg(WheellegEnvCfg):
    """WheelLeg timed jump task with fixed training settings."""

    actions: JumpActionsCfg = JumpActionsCfg()
    observations: JumpObservationsCfg = JumpObservationsCfg()
    commands: JumpCommandsCfg = JumpCommandsCfg()
    rewards: JumpRewardsCfg = JumpRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 6.0
        self.scene.robot.actuators = {
            "hips": ImplicitActuatorCfg(
                joint_names_expr=[
                    "L_hip_front_joint",
                    "L_hip_rear_joint",
                    "R_hip_front_joint",
                    "R_hip_rear_joint",
                ],
                effort_limit_sim=24.0,
                velocity_limit_sim=14.0,
                stiffness=30.0,
                damping=1.0,
            ),
            "passive_knees": ImplicitActuatorCfg(
                joint_names_expr=[
                    "L_knee_front_joint",
                    "L_knee_rear_joint",
                    "R_knee_front_joint",
                    "R_knee_rear_joint",
                ],
                effort_limit_sim=0.5,
                velocity_limit_sim=10.0,
                stiffness=0.0,
                damping=0.2,
            ),
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=[
                    "L_wheel_joint",
                    "R_wheel_joint",
                ],
                effort_limit_sim=0.8,
                velocity_limit_sim=31.416,
                stiffness=1.0,
                damping=0.2,
            ),
        }
