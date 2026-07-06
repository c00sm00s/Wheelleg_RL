# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Wheel-legged robot scene configuration for Isaac Lab."""

import math
from pathlib import Path


import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass


from . import mdp

##
# Pre-defined configs
##

# from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip


##
# Scene definition
##
ROBOT_USD_PATH = Path(__file__).resolve().parents[3].joinpath(
    "assets",
    "WheelLeg-USD",
    "WheelLeg-USD.usd"
)


@configclass
class WheellegSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ROBOT_USD_PATH),
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[
                    "L_hip_front_joint",
                    "L_knee_front_joint",
                    "L_hip_rear_joint",
                    "L_knee_rear_joint",
                    "R_hip_front_joint",
                    "R_knee_front_joint",
                    "R_hip_rear_joint",
                    "R_knee_rear_joint",
                ],
                effort_limit_sim=9.0,
                velocity_limit_sim=5.236,
                stiffness=20.0,
                damping=1.0,
            ),
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=[
                    "L_wheel_joint",
                    "R_wheel_joint",
                ],
                effort_limit_sim=0.9,
                velocity_limit_sim=31.416,
                stiffness=1.0,
                damping=0.2,
            ),
        },
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    # contact sensors
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

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
class CommandsCfg:
    """Command specifications for the MDP."""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 8.0),
        rel_standing_envs=0.2,
        rel_heading_envs=1.0,
        heading_command=True,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi)
        )
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        Velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"}
        )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-0.2, 0.2)},
            "velocity_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.02, 0.02),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            },
        },)


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_exp,
                                   weight=1.5,
                                   params={
                                       "command_name": "base_velocity",
                                       "std": 0.35})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_exp,
                                  weight=1.0,
                                  params={
                                      "command_name": "base_velocity",
                                      "std": 0.35})
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_torque_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_too_low = DoneTerm(func=mdp.root_height_below_minimum,
                            params={"minimum_height": 0.15})
    bad_orientation = DoneTerm(func=mdp.bad_orientation,
                               params={"limit_angle": 0.9})
    base_contact = DoneTerm(func=mdp.illegal_contact,
                            params={"sensor_cfg": SceneEntityCfg(
                                "contact_sensor",
                                body_names="base_link",
                            ),
                                "threshold": 1.0
                            },
                            )
##
# Environment configuration
##


@configclass
class WheellegEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: WheellegSceneCfg = WheellegSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
