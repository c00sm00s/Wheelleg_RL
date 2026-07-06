# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Rough-terrain locomotion task for the WheelLeg robot.

The file name keeps the existing project spelling (``Terrian``), but the task is
a terrain locomotion environment. It follows the Unitree/Isaac Lab terrain
curriculum pattern: start on easy terrain rows, move successful envs to harder
rows, and slowly widen velocity commands after tracking becomes reliable.
"""

from __future__ import annotations

import math

import numpy as np
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import isaaclab.terrains.trimesh.mesh_terrains as mesh_terrain_fns
import trimesh
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .mdp.rewards import (
    five_bar_singularity_penalty,
    joint_velocity_limit_penalty,
    leg_joint_limit_margin_penalty,
    leg_joint_motion_reward,
    wheel_clearance_difference_penalty,
    gait_alternating_contact_reward,
    gait_both_legs_participation_reward,
    gait_lateral_drift_penalty,
    gait_swing_clearance_reward,
    gait_swing_drag_penalty,
    history_blocked_drag_penalty,
    history_blocked_lift_velocity_reward,
    history_blocked_swing_clearance_reward,
    safe_base_height_l2,
    stage_blocked_asymmetric_wheel_lift_reward,
    stage_track_ang_vel_z_exp,
    stage_track_lin_vel_xy_exp,
    stage3_heading_alignment_reward,
    stage3_split_stair_lagging_lift_reward,
    stage3_stair_progress_reward,
    stance_wheel_lateral_slip_penalty,
    wheel_not_stop_penalty,
    wheel_rolling_consistency,
)
from .wheelleg_env_cfg import ROBOT_USD_PATH, RewardsCfg as BaseRewardsCfg, TerminationsCfg, WheellegEnvCfg


WHEEL_JOINT_CFG = SceneEntityCfg("robot", joint_names=["L_wheel_joint", "R_wheel_joint"])
HIP_JOINT_CFG = SceneEntityCfg(
    "robot",
    joint_names=["L_hip_front_joint", "L_hip_rear_joint", "R_hip_front_joint", "R_hip_rear_joint"],
)
KNEE_JOINT_CFG = SceneEntityCfg(
    "robot",
    joint_names=["L_knee_front_joint", "L_knee_rear_joint", "R_knee_front_joint", "R_knee_rear_joint"],
)
ACTUATED_JOINT_CFG = SceneEntityCfg(
    "robot",
    joint_names=[
        "L_hip_front_joint",
        "L_hip_rear_joint",
        "R_hip_front_joint",
        "R_hip_rear_joint",
        "L_wheel_joint",
        "R_wheel_joint",
    ],
)
BASE_BODY_CFG = SceneEntityCfg("robot", body_names=["base_link"])
WHEEL_BODY_CFG = SceneEntityCfg("robot", body_names=["L_wheel_link", "R_wheel_link"])
WHEEL_CONTACT_CFG = SceneEntityCfg("contact_sensor", body_names=["L_wheel_link", "R_wheel_link"])
LEG_LINK_CONTACT_CFG = SceneEntityCfg(
    "contact_sensor",
    body_names=[
        "L_thigh_front_link",
        "L_calf_front_link",
        "L_thigh_rear_link",
        "L_calf_rear_link",
        "R_thigh_front_link",
        "R_calf_front_link",
        "R_thigh_rear_link",
        "R_calf_rear_link",
    ],
)
STAGE2_INDEX = 2
# Final straight-stair curriculum column. The old down-stair/pyramid stage was
# removed, so the straight stair is now column 3 in a four-stage curriculum.
STAGE3_INDEX = 3


def raised_inverted_pyramid_stairs_terrain(difficulty: float, cfg):
    """Generate up-stairs terrain whose lowest center platform is at z=0.

    Isaac Lab's inverted pyramid stairs are centered below the world plane: the
    returned origin is negative, which makes ``root_height_below_minimum`` kill
    the robot as soon as it reaches the bottom. This wrapper shifts the whole
    mesh upward by that negative origin so the bottom is 0 m and all stairs are
    above the ground plane.
    """

    meshes, origin = mesh_terrain_fns.inverted_pyramid_stairs_terrain(difficulty, cfg)
    height_offset = -float(origin[2])
    for mesh in meshes:
        mesh.apply_translation((0.0, 0.0, height_offset))
    origin = origin.copy()
    origin[2] = 0.0
    return meshes, origin


@configclass
class MeshRaisedInvertedPyramidStairsTerrainCfg(terrain_gen.MeshInvertedPyramidStairsTerrainCfg):
    """Inverted stairs shifted upward so stage2 never starts below world z=0."""

    function = raised_inverted_pyramid_stairs_terrain


def straight_stairs_terrain(difficulty: float, cfg):
    """Generate a one-way +X stair course with the lowest walking surface at z=0."""

    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    terrain_center_y = 0.5 * cfg.size[1]
    x0 = cfg.border_width + cfg.bottom_platform_width
    meshes = [mesh_terrain_fns.make_plane(cfg.size, 0.0, center_zero=False)]

    # Each box top is the walking surface. The bottom extends slightly below
    # zero so all stair faces are closed meshes while the lowest surface remains
    # at world height 0 after terrain placement.
    base_thickness = cfg.base_thickness
    for step_idx in range(1, cfg.num_steps + 1):
        top_height = step_idx * step_height
        box_dims = (cfg.step_width, cfg.stair_width, top_height + base_thickness)
        box_pos = (
            x0 + (step_idx - 0.5) * cfg.step_width,
            terrain_center_y,
            0.5 * (top_height - base_thickness),
        )
        meshes.append(trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos)))

    top_height = cfg.num_steps * step_height
    top_dims = (cfg.top_platform_width, cfg.stair_width, top_height + base_thickness)
    top_pos = (
        x0 + cfg.num_steps * cfg.step_width + 0.5 * cfg.top_platform_width,
        terrain_center_y,
        0.5 * (top_height - base_thickness),
    )
    meshes.append(trimesh.creation.box(top_dims, trimesh.transformations.translation_matrix(top_pos)))

    origin = np.array([cfg.border_width + 0.5 * cfg.bottom_platform_width, terrain_center_y, 0.0])
    return meshes, origin


@configclass
class MeshStraightStairsTerrainCfg(terrain_gen.SubTerrainBaseCfg):
    """Single-direction stair terrain for stage3.

    Tune these fields to change the human-style stair shape:
    - ``step_height_range``: stair riser height. Use equal values for fixed height.
    - ``step_width``: stair tread depth along +X.
    - ``num_steps``: number of risers before the top platform.
    - ``stair_width``: usable width in Y; leaving it should count as falling off.
    """

    function = straight_stairs_terrain

    step_height_range: tuple[float, float] = (0.08, 0.08)
    step_width: float = 0.35
    num_steps: int = 10
    stair_width: float = 2.0
    bottom_platform_width: float = 2.0
    top_platform_width: float = 2.0
    border_width: float = 1.0
    base_thickness: float = 0.20


STAGE3_STAIRS_CFG = MeshStraightStairsTerrainCfg(
    proportion=1.0,
    step_height_range=(0.08, 0.08),
    step_width=0.35,
    num_steps=10,
    stair_width=5.0,
    bottom_platform_width=4.0,
    top_platform_width=3.0,
    border_width=0.5,
)
"""Stage3 exposed stair interface.

Edit this object to tune the straight stair shape. Termination and command
logic below reference the same values, so stair geometry and task bounds stay
consistent.
"""


WHEELLEG_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    # Four-stage curriculum with one row and four terrain-type columns.
    # col 0 -> flat, col 1 -> rough, col 2 -> up stairs,
    # col 3 -> single-direction human-style up stairs.
    # Isaac Lab uses rows as difficulty levels and columns as terrain types;
    # because we want exactly one difficulty per stage, this is 1x4 instead of 4x1.
    size=(16.0, 16.0),
    border_width=20.0,
    num_rows=1,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        # Stage 0: learn stable velocity tracking and balance on flat ground.
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
        # Stage 1: add low-height unevenness after flat locomotion works.
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(0.01, 0.08),
            noise_step=0.01,
            border_width=0.25,
        ),
        # Stage 2: start on a 0 m center platform and go outward/upstairs.
        # The custom cfg shifts Isaac Lab's inverted stairs upward, so the
        # lowest point is not below world z=0 and base_too_low stays meaningful.
        "up_stairs": MeshRaisedInvertedPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.015, 0.045),
            step_width=0.90,
            platform_width=6.0,
            border_width=1.0,
            holes=False,
        ),
        # Stage 3: human-style straight stairs. Lowest walking surface is 0 m,
        # and the only valid success direction is world +X.
        "human_straight_up_stairs": STAGE3_STAIRS_CFG,
    },
)


@configclass
class WheellegTerrianSceneCfg(InteractiveSceneCfg):
    """Scene with generated terrain, WheelLeg robot, and terrain sensors."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=WHEELLEG_TERRAINS_CFG,
        # Start every env on row 0 / col 0, i.e. flat ground. The curriculum
        # promotes successful envs through rough, up-stair, and straight-stair stages.
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(ROBOT_USD_PATH), activate_contact_sensors=True),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
        actuators={
            "hips": ImplicitActuatorCfg(
                joint_names_expr=[
                    "L_hip_front_joint",
                    "L_hip_rear_joint",
                    "R_hip_front_joint",
                    "R_hip_rear_joint",
                ],
                effort_limit_sim=12.0,
                velocity_limit_sim=5.236,
                stiffness=20.0,
                damping=1.0,
            ),
            "passive_knees": ImplicitActuatorCfg(
                joint_names_expr=[
                    "L_knee_front_joint",
                    "L_knee_rear_joint",
                    "R_knee_front_joint",
                    "R_knee_rear_joint",
                ],
                effort_limit_sim=12.0,
                velocity_limit_sim=5.236,
                stiffness=0.0,
                damping=0.3,
            ),
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=["L_wheel_joint", "R_wheel_joint"],
                effort_limit_sim=0.9,
                velocity_limit_sim=31.416,
                stiffness=1.0,
                damping=0.2,
            ),
        },
    )

    # Height samples in front/around the base give the policy terrain context.
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # Privileged foot-local terrain samples for the critic/teacher. These
    # sensors are intentionally not part of the deployable policy observation.
    left_foot_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/L_wheel_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.6, 0.6)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    right_foot_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/R_wheel_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.6, 0.6)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class TerrianActionsCfg:
    """Actions for rough-terrain locomotion."""

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
class TerrianCommandsCfg:
    """Velocity commands with small initial range and larger curriculum target."""

    base_velocity = mdp.StageAwareLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            # Stage0-2 sample random body-frame velocity commands at reset and
            # hold them for the full episode. Stage3 overrides this with a
            # world-frame +X uphill command inside StageAwareLevelVelocityCommand.
            lin_vel_x=(-0.80, 0.80),
            lin_vel_y=(-0.35, 0.35),
            ang_vel_z=(-0.50, 0.50),
            heading=(-math.pi, math.pi),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            # Command curriculum may widen the stage0-2 random command range.
            lin_vel_x=(-1.5, 1.5),
            lin_vel_y=(-0.8, 0.8),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
        stage3_index=STAGE3_INDEX,
        stage3_world_direction=(1.0, 0.0),
        stage3_lin_vel_x=(0.8, 1.5),
        stage3_heading_control_stiffness=1.5,
        stage3_ang_vel_z=(-0.1, 0.1),
    )


@configclass
class TerrianObservationsCfg:
    """Policy observations for velocity tracking over uneven terrain."""

    @configclass
    class PolicyCfg(ObsGroup):
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.10, n_max=0.10))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.03, n_max=0.03))
        # Five-bar mechanism state: drive motor angles/velocities plus the
        # resolved wheel/end-effector pose and velocity in the base frame.
        drive_motor_state = ObsTerm(
            func=mdp.drive_motor_state_obs,
            params={"asset_cfg": HIP_JOINT_CFG},
        )
        wheel_end_effector_state = ObsTerm(
            func=mdp.wheel_end_effector_state_obs,
            params={"asset_cfg": WHEEL_BODY_CFG},
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-0.5, n_max=0.5))
        last_action = ObsTerm(func=mdp.last_action)
        # Recent velocity tracking error lets the policy detect a sudden block:
        # small error while rolling, then a sharp spike when the base is stopped.
        velocity_error_history = ObsTerm(
            func=mdp.velocity_tracking_error_history_obs,
            params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot"), "history_length": 5},
        )
        # Global blind-walking memory. This lets the policy connect recent
        # actuator changes with blocked-wheel and lift outcomes.
        action_history = ObsTerm(func=mdp.action_history_obs, params={"history_length": 5})
        stage2_asymmetric_gait_phase = ObsTerm(
            func=mdp.stage_asymmetric_gait_phase_obs,
            params={
                "stage": STAGE2_INDEX,
                "gait_period": 0.75,
                "swing_fraction": 0.35,
                "blocked_latch_attr": "wheelleg_blocked_wheel_latch_min_stage_2",
                "state_prefix": "wheelleg_stage2_blocked_asymmetric_gait",
            },
        )
        gait_phase = None
        wheel_contact = ObsTerm(
            func=mdp.contact_state_obs,
            params={"sensor_cfg": WHEEL_CONTACT_CFG, "threshold": 1.0},
        )
        leg_link_contact = ObsTerm(
            func=mdp.leg_link_contact_state_obs,
            params={"sensor_cfg": LEG_LINK_CONTACT_CFG, "threshold": 1.0, "links_per_side": 4},
        )
        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        left_terrain = ObsTerm(
            func=mdp.foot_height_scan,
            params={"sensor_cfg": SceneEntityCfg("left_foot_scanner"), "offset": 0.5},
            clip=(-1.0, 1.0),
        )
        right_terrain = ObsTerm(
            func=mdp.foot_height_scan,
            params={"sensor_cfg": SceneEntityCfg("right_foot_scanner"), "offset": 0.5},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class TerrianEventCfg:
    """Domain randomization and reset events for robust terrain learning."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform_stage_yaw,
        mode="reset",
        params={
            # Sample inside the selected terrain tile instead of always starting
            # at the exact origin. Stage3 keeps yaw aligned with the stair.
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
            "stage_pose_ranges": {
                # Stage2 is an outward-up inverted stair field. Keep resets near
                # the enlarged center platform so the first ring is not touched
                # immediately after reset.
                str(STAGE2_INDEX): {"x": (-0.35, 0.35), "y": (-0.35, 0.35)},
                # Stage3 is a +X straight stair course. Reset on the rear half
                # of the bottom platform, well before the first riser.
                str(STAGE3_INDEX): {"x": (-0.5, -0.5), "y": (-0.25, 0.25), "yaw": (-0.1, 0.1)},
            },
            "velocity_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.02, 0.02),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            },
            "stage": STAGE3_INDEX,
            "stage_yaw_range": (-0.1, 0.1),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.9, 1.1), "velocity_range": (-0.2, 0.2)},
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(6.0, 10.0),
        params={"velocity_range": {"x": (-0.25, 0.25), "y": (-0.20, 0.20)}},
    )


@configclass
class TerrianRewardsCfg(BaseRewardsCfg):
    """Reward terms tuned for rolling locomotion on generated terrain."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=3.0,
        params={"command_name": "base_velocity", "std": 0.45},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.2,
        params={"command_name": "base_velocity", "std": 0.45},
    )
    stage2plus_track_lin_vel_xy_exp = RewTerm(
        func=stage_track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "min_stage": STAGE2_INDEX, "std": 0.45},
    )
    stage2plus_track_ang_vel_z_exp = RewTerm(
        func=stage_track_ang_vel_z_exp,
        weight=0.8,
        params={"command_name": "base_velocity", "min_stage": STAGE2_INDEX, "std": 0.45},
    )
    alive = RewTerm(func=mdp.is_alive, weight=0.5)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    # Keep the body upright, but do not over-penalize vertical motion: stairs
    # require the base to rise/fall while the legs absorb terrain height changes.
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.50)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.08)
    base_height_l2 = RewTerm(
        func=safe_base_height_l2,
        weight=-0.50,
        params={"target_height": 0.50, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    joint_torque_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": ACTUATED_JOINT_CFG},
    )
    joint_power_l1 = RewTerm(
        func=mdp.joint_power_l1,
        weight=-2.0e-6,
        params={"asset_cfg": ACTUATED_JOINT_CFG},
    )
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": ACTUATED_JOINT_CFG},
    )
    base_lin_acc_l2 = RewTerm(
        func=mdp.body_lin_acc_l2,
        weight=-2.0e-4,
        params={"asset_cfg": BASE_BODY_CFG},
    )
    joint_velocity_limits = RewTerm(
        func=joint_velocity_limit_penalty,
        weight=-0.08,
        params={"asset_cfg": ACTUATED_JOINT_CFG, "soft_ratio": 0.90},
    )

    # Suspension-style behavior: on rough/stair stages, reward moderate hip
    # motion only when velocity tracking and base attitude are already good.
    leg_motion = RewTerm(
        func=leg_joint_motion_reward,
        weight=0.05,
        params={"command_name": "base_velocity", "asset_cfg": HIP_JOINT_CFG, "min_stage": 1},
    )
    # Avoid learning the stiff-strut solution where hips sit near mechanical
    # limits and the robot drops down stairs with locked legs.
    leg_limit_margin = RewTerm(
        func=leg_joint_limit_margin_penalty,
        weight=-0.12,
        params={"asset_cfg": HIP_JOINT_CFG, "margin": 0.80},
    )
    # Passive knee joints close to either end of travel are near folded/straight
    # five-bar singular configurations. Penalize the outer 25% steeply so the
    # linkage keeps room to absorb terrain contact instead of locking.
    five_bar_singularity = RewTerm(
        func=five_bar_singularity_penalty,
        weight=-0.20,
        params={"asset_cfg": KNEE_JOINT_CFG, "margin": 0.75, "exponent": 4.0, "max_penalty": 4.0},
    )
    wheel_clearance_difference = RewTerm(
        func=wheel_clearance_difference_penalty,
        weight=-0.25,
        params={
            "asset_cfg": WHEEL_BODY_CFG,
            "left_sensor_cfg": SceneEntityCfg("left_foot_scanner"),
            "right_sensor_cfg": SceneEntityCfg("right_foot_scanner"),
            "wheel_radius": 0.0625,
            "min_stage": STAGE2_INDEX,
            "deadband": 0.16,
            "std": 0.05,
        },
    )

    # Mild trot-style gait shaping. Keep these weaker than the terrain/blocking
    # rewards so the robot can still break rhythm when climbing obstacles.
    gait_alternating_contact = RewTerm(
        func=gait_alternating_contact_reward,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": WHEEL_CONTACT_CFG,
            "gait_period": 0.75,
            "double_support_width": 0.14,
        },
    )
    gait_swing_clearance = None
    gait_swing_drag = RewTerm(
        func=gait_swing_drag_penalty,
        weight=-0.20,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": WHEEL_CONTACT_CFG,
            "gait_period": 0.75,
            "force_scale": 20.0,
        },
    )
    gait_both_legs_participation = None
    gait_lateral_drift = RewTerm(
        func=gait_lateral_drift_penalty,
        weight=-0.3,
        params={"command_name": "base_velocity"},
    )
    undesired_leg_contacts = RewTerm(
        func=mdp.undesired_contacts_count,
        weight=-1.5,
        params={"sensor_cfg": LEG_LINK_CONTACT_CFG, "threshold": 1.0},
    )
    stance_wheel_lateral_slip = None
    # History-triggered obstacle negotiation. Stage2 uses the asymmetric
    # blocked gait below; stage3 clearance is measured as wheel-bottom height
    # above foot-local ray terrain instead of analytic stair geometry.
    history_blocked_swing_clearance = RewTerm(
        func=history_blocked_swing_clearance_reward,
        weight=1.25,
        params={
            "command_name": "base_velocity",
            "asset_cfg": WHEEL_BODY_CFG,
            "sensor_cfg": WHEEL_CONTACT_CFG,
            "link_sensor_cfg": LEG_LINK_CONTACT_CFG,
            "gait_period": 0.8,
            "history_length": 5,
            "lift_height": 0.22,
            "wheel_radius": 0.0625,
            "left_sensor_cfg": SceneEntityCfg("left_foot_scanner"),
            "right_sensor_cfg": SceneEntityCfg("right_foot_scanner"),
            "stage": STAGE3_INDEX,
            "bottom_platform_width": STAGE3_STAIRS_CFG.bottom_platform_width,
            "step_height": STAGE3_STAIRS_CFG.step_height_range[0],
            "step_width": STAGE3_STAIRS_CFG.step_width,
            "num_steps": STAGE3_STAIRS_CFG.num_steps,
            "wheel_velocity_error_threshold": 0.25,
            "horizontal_force_ratio_threshold": 0.4,
            "horizontal_force_threshold": 1.0,
            "wheel_spin_speed_threshold": 0.5,
            "wheel_spin_forward_speed_threshold": 0.12,
            "min_blocked_stage": STAGE3_INDEX,
        },
    )
    stage2_blocked_asymmetric_lift = RewTerm(
        func=stage_blocked_asymmetric_wheel_lift_reward,
        weight=0.75,
        params={
            "command_name": "base_velocity",
            "stage": STAGE2_INDEX,
            "asset_cfg": WHEEL_BODY_CFG,
            "sensor_cfg": WHEEL_CONTACT_CFG,
            "link_sensor_cfg": LEG_LINK_CONTACT_CFG,
            "left_sensor_cfg": SceneEntityCfg("left_foot_scanner"),
            "right_sensor_cfg": SceneEntityCfg("right_foot_scanner"),
            "gait_period": 0.75,
            "swing_fraction": 0.35,
            "history_length": 5,
            "min_command_speed": 0.1,
            "contact_threshold": 1.0,
            "wheel_velocity_error_threshold": 0.25,
            "horizontal_force_ratio_threshold": 0.4,
            "horizontal_force_threshold": 1.0,
            "wheel_spin_speed_threshold": 0.5,
            "wheel_spin_forward_speed_threshold": 0.12,
            "height_margin": 0.06,
            "height_std": 0.03,
            "wheel_radius": 0.0625,
            "min_base_height": 0.42,
            "base_height_std": 0.05,
            "min_forward_speed": 0.08,
            "max_lift_height": 0.16,
            "max_wheel_clearance_difference": 0.20,
            "height_weight": 1.0,
            "air_weight": 0.7,
            "support_contact_weight": 0.3,
            "state_prefix": "wheelleg_stage2_blocked_asymmetric_gait",
        },
    )
    history_blocked_lift_velocity = RewTerm(
        func=history_blocked_lift_velocity_reward,
        weight=0.35,
        params={
            "command_name": "base_velocity",
            "asset_cfg": WHEEL_BODY_CFG,
            "sensor_cfg": WHEEL_CONTACT_CFG,
            "link_sensor_cfg": LEG_LINK_CONTACT_CFG,
            "history_length": 5,
            "target_lift_speed": 0.50,
            "wheel_velocity_error_threshold": 0.25,
            "horizontal_force_ratio_threshold": 0.4,
            "horizontal_force_threshold": 1.0,
            "wheel_spin_radius": 0.0625,
            "wheel_spin_speed_threshold": 0.5,
            "wheel_spin_forward_speed_threshold": 0.12,
            "min_blocked_stage": 2,
        },
    )
    history_blocked_drag = RewTerm(
        func=history_blocked_drag_penalty,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": WHEEL_BODY_CFG,
            "sensor_cfg": WHEEL_CONTACT_CFG,
            "link_sensor_cfg": LEG_LINK_CONTACT_CFG,
            "gait_period": 0.8,
            "history_length": 5,
            "wheel_velocity_error_threshold": 0.25,
            "horizontal_force_ratio_threshold": 0.4,
            "horizontal_force_threshold": 1.0,
            "wheel_spin_radius": 0.0625,
            "wheel_spin_speed_threshold": 0.5,
            "wheel_spin_forward_speed_threshold": 0.12,
            "min_blocked_stage": 2,
        },
    )
    stage3_stair_progress = RewTerm(
        func=stage3_stair_progress_reward,
        weight=2.0,
        params={
            "stage": STAGE3_INDEX,
            "bottom_platform_width": STAGE3_STAIRS_CFG.bottom_platform_width,
            "step_height": STAGE3_STAIRS_CFG.step_height_range[0],
            "step_width": STAGE3_STAIRS_CFG.step_width,
            "num_steps": STAGE3_STAIRS_CFG.num_steps,
            "asset_cfg": WHEEL_BODY_CFG,
            "wheel_radius": 0.0625,
            "height_margin": 0.04,
        },
    )
    stage3_split_stair_lagging_lift = RewTerm(
        func=stage3_split_stair_lagging_lift_reward,
        weight=1.0,
        params={
            "stage": STAGE3_INDEX,
            "bottom_platform_width": STAGE3_STAIRS_CFG.bottom_platform_width,
            "step_height": STAGE3_STAIRS_CFG.step_height_range[0],
            "step_width": STAGE3_STAIRS_CFG.step_width,
            "num_steps": STAGE3_STAIRS_CFG.num_steps,
            "asset_cfg": WHEEL_BODY_CFG,
            "left_sensor_cfg": SceneEntityCfg("left_foot_scanner"),
            "right_sensor_cfg": SceneEntityCfg("right_foot_scanner"),
            "wheel_radius": 0.0625,
            "height_margin": 0.04,
            "lift_height": 0.16,
            "std": 0.05,
        },
    )
    stage3_heading_alignment = RewTerm(
        func=stage3_heading_alignment_reward,
        weight=0.5,
        params={"stage": STAGE3_INDEX, "target_heading": 0.0, "std": 0.35},
    )

    wheel_rolling = RewTerm(
        func=wheel_rolling_consistency,
        weight=0.35,
        params={"command_name": "base_velocity", "wheel_radius": 0.0625, "asset_cfg": WHEEL_JOINT_CFG, "std": 0.45},
    )
    wheel_stop = RewTerm(
        func=wheel_not_stop_penalty,
        weight=-0.30,
        params={"command_name": "base_velocity", "asset_cfg": WHEEL_JOINT_CFG},
    )


@configclass
class TerrianCurriculumCfg:
    """Curriculum terms following the Unitree terrain curriculum idea."""

    # Promotes/demotes each env across terrain columns based on timeout success.
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    # Expands velocity command ranges only after the tracking reward is high.
    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)
    ang_vel_cmd_levels = CurrTerm(func=mdp.ang_vel_cmd_levels)


@configclass
class TerrianTerminationsCfg(TerminationsCfg):
    """Terrain task termination terms.

    Stage3 top completion is marked as a timeout-like success so the terrain
    curriculum does not treat reaching the stair top as a failure.
    """

    stage3_reached_top = DoneTerm(
        func=mdp.stage3_reached_top,
        time_out=True,
        params={
            "stage": STAGE3_INDEX,
            "bottom_platform_width": STAGE3_STAIRS_CFG.bottom_platform_width,
            "step_width": STAGE3_STAIRS_CFG.step_width,
            "num_steps": STAGE3_STAIRS_CFG.num_steps,
            "stair_width": STAGE3_STAIRS_CFG.stair_width,
            "fall_margin": 0.35,
            "top_success_margin": 0.50,
        },
    )
    stage3_fell_off_stairs = DoneTerm(
        func=mdp.stage3_fell_off_stairs,
        params={
            "stage": STAGE3_INDEX,
            "wheel_body_cfg": WHEEL_BODY_CFG,
            "bottom_platform_width": STAGE3_STAIRS_CFG.bottom_platform_width,
            "step_height": STAGE3_STAIRS_CFG.step_height_range[0],
            "step_width": STAGE3_STAIRS_CFG.step_width,
            "num_steps": STAGE3_STAIRS_CFG.num_steps,
            "stair_width": STAGE3_STAIRS_CFG.stair_width,
            "fall_margin": 0.35,
            "top_success_margin": 0.50,
            "height_check_start_step": 2,
            # Wheel diameter is 0.125 m, so wheel bottom is wheel_link.z - 0.0625.
            # Use wheel-bottom height instead of base height to avoid resetting
            # normal stair-climb attempts where the base reaches forward before
            # the wheels have climbed the riser.
            "wheel_radius": 0.0625,
            "wheel_height_margin": 0.03,
            "require_both_wheels_below": True,
            "enable_height_fall_check": True,
        },
    )
    stage3_base_contact = DoneTerm(
        func=mdp.stage3_base_stair_contact,
        params={
            "stage": STAGE3_INDEX,
            "bottom_platform_width": STAGE3_STAIRS_CFG.bottom_platform_width,
            "step_height": STAGE3_STAIRS_CFG.step_height_range[0],
            "step_width": STAGE3_STAIRS_CFG.step_width,
            "num_steps": STAGE3_STAIRS_CFG.num_steps,
            "stair_width": STAGE3_STAIRS_CFG.stair_width,
            "contact_margin": 0.01,
            "base_half_length": 0.115,
            "base_half_width": 0.09,
            "base_bottom_z_offset": -0.03,
            "lateral_margin": 0.05,
        },
    )
    stage3_split_stair_stance_timeout = DoneTerm(
        func=mdp.stage3_split_stair_stance_timeout,
        params={
            "stage": STAGE3_INDEX,
            "wheel_body_cfg": WHEEL_BODY_CFG,
            "bottom_platform_width": STAGE3_STAIRS_CFG.bottom_platform_width,
            "step_height": STAGE3_STAIRS_CFG.step_height_range[0],
            "step_width": STAGE3_STAIRS_CFG.step_width,
            "num_steps": STAGE3_STAIRS_CFG.num_steps,
            "wheel_radius": 0.0625,
            "stance_height_margin": 0.04,
            "max_duration_s": 1.2,
        },
    )
    stage3_heading_error_timeout = DoneTerm(
        func=mdp.stage3_heading_error_timeout,
        params={
            "stage": STAGE3_INDEX,
            "target_heading": 0.0,
            "max_heading_error": math.radians(60.0),
            "max_duration_s": 1.0,
        },
    )


@configclass
class WheellegTerrianEnvCfg(WheellegEnvCfg):
    """WheelLeg locomotion environment over generated terrain."""

    scene: WheellegTerrianSceneCfg = WheellegTerrianSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: TerrianObservationsCfg = TerrianObservationsCfg()
    actions: TerrianActionsCfg = TerrianActionsCfg()
    commands: TerrianCommandsCfg = TerrianCommandsCfg()
    rewards: TerrianRewardsCfg = TerrianRewardsCfg()
    terminations: TerrianTerminationsCfg = TerrianTerminationsCfg()
    events: TerrianEventCfg = TerrianEventCfg()
    curriculum: TerrianCurriculumCfg = TerrianCurriculumCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 20.0
        self.scene.contact_sensor.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.left_foot_scanner.update_period = self.decimation * self.sim.dt
        self.scene.right_foot_scanner.update_period = self.decimation * self.sim.dt
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.viewer.eye = (-4.0, 30.0, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)

        # Terrain curriculum must be enabled on the generator itself; the
        # curriculum manager only calls ``terrain_levels_vel`` at reset.
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class WheellegTerrianPlayEnvCfg(WheellegTerrianEnvCfg):
    """Small terrain layout for quick visual checks and policy playback."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 4
        self.scene.terrain.max_init_terrain_level = 0
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
