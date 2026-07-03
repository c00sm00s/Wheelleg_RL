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

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import isaaclab.terrains.trimesh.mesh_terrains as mesh_terrain_fns
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
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
    gait_alternating_contact_reward,
    gait_both_legs_participation_reward,
    gait_lateral_drift_penalty,
    gait_swing_clearance_reward,
    gait_swing_drag_penalty,
    safe_base_height_l2,
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
WHEEL_BODY_CFG = SceneEntityCfg("robot", body_names=["L_wheel_link", "R_wheel_link"])
WHEEL_CONTACT_CFG = SceneEntityCfg("contact_sensor", body_names=["L_wheel_link", "R_wheel_link"])
LEFT_WHEEL_SCANNER_CFG = SceneEntityCfg("left_wheel_scanner")
RIGHT_WHEEL_SCANNER_CFG = SceneEntityCfg("right_wheel_scanner")


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
    """Inverted stairs shifted upward so stage3 never starts below world z=0."""

    function = raised_inverted_pyramid_stairs_terrain


WHEELLEG_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    # Four-stage curriculum with one row and four terrain-type columns.
    # col 0 -> flat, col 1 -> rough, col 2 -> down stairs, col 3 -> up stairs.
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
        # Stage 2: start on the high center platform and go outward/downstairs.
        # Larger step width keeps the total stair height reasonable on 16 m tiles.
        "down_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.015, 0.055),
            step_width=0.80,
            platform_width=4.0,
            border_width=1.0,
            holes=False,
        ),
        # Stage 3: start on a 0 m center platform and go outward/upstairs.
        # The custom cfg shifts Isaac Lab's inverted stairs upward, so the
        # lowest point is not below world z=0 and base_too_low stays meaningful.
        "up_stairs": MeshRaisedInvertedPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.015, 0.055),
            step_width=0.80,
            platform_width=4.0,
            border_width=1.0,
            holes=False,
        ),
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
        # promotes successful envs through rough, down-stair, and up-stair stages.
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
                joint_names_expr=["L_wheel_joint", "R_wheel_joint"],
                effort_limit_sim=0.9,
                velocity_limit_sim=31.416,
                stiffness=0.0,
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

    # Local ground probes below each wheel. Stage3 swing rewards compare wheel
    # height against these hits, so stair clearance is terrain-relative.
    left_wheel_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/L_wheel_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.25)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.02, size=[0.02, 0.02]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    right_wheel_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/R_wheel_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.25)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.02, size=[0.02, 0.02]),
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

    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[
            "L_hip_front_joint",
            "L_hip_rear_joint",
            "R_hip_front_joint",
            "R_hip_rear_joint",
            "L_wheel_joint",
            "R_wheel_joint",
        ],
        # Use most available hip authority for obstacle negotiation while keeping
        # wheel effort bounded by the actuator limit.
        scale={".*hip.*": 6.0, ".*wheel.*": 0.9},
    )


@configclass
class TerrianCommandsCfg:
    """Velocity commands with small initial range and larger curriculum target."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            # Sample one forward speed at reset and keep it for the full episode.
            # Lateral/yaw commands are fixed to zero so the robot cannot turn
            # around or sidestep away from the stair task.
            lin_vel_x=(0.20, 1.20),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(-math.pi, math.pi),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            # Command curriculum may widen only the forward-speed range.
            lin_vel_x=(0.20, 1.20),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(-math.pi, math.pi),
        ),
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
        action_history = ObsTerm(func=mdp.action_history_obs, params={"history_length": 5})
        gait_phase = ObsTerm(func=mdp.gait_phase_obs, params={"gait_period": 0.8})
        wheel_contact = ObsTerm(
            func=mdp.contact_state_obs,
            params={"sensor_cfg": WHEEL_CONTACT_CFG, "threshold": 1.0},
        )
        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerrianEventCfg:
    """Domain randomization and reset events for robust terrain learning."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # Sample inside the selected terrain tile instead of always starting
            # at the exact origin. This exposes the policy to local roughness.
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.02, 0.02),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            },
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
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.35},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": 0.35},
    )
    alive = RewTerm(func=mdp.is_alive, weight=0.5)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)
    # Keep the body upright, but do not over-penalize vertical motion: stairs
    # require the base to rise/fall while the legs absorb terrain height changes.
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.50)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.08)
    base_height_l2 = RewTerm(
        func=safe_base_height_l2,
        weight=-0.80,
        params={"target_height": 0.50, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.012)
    joint_torque_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": ACTUATED_JOINT_CFG},
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

    # Global gait rewards: force a left/right swing-stance pattern on every
    # terrain stage instead of waiting until stage3 to introduce leg lifting.
    gait_alternating_contact = RewTerm(
        func=gait_alternating_contact_reward,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": WHEEL_CONTACT_CFG,
            "gait_period": 0.8,
            "double_support_width": 0.18,
        },
    )
    gait_swing_clearance = RewTerm(
        func=gait_swing_clearance_reward,
        weight=0.8,
        params={
            "command_name": "base_velocity",
            "asset_cfg": WHEEL_BODY_CFG,
            "left_sensor_cfg": LEFT_WHEEL_SCANNER_CFG,
            "right_sensor_cfg": RIGHT_WHEEL_SCANNER_CFG,
            "gait_period": 0.8,
            "clearance_height": 0.14,
            "wheel_radius": 0.10,
        },
    )
    gait_swing_drag = RewTerm(
        func=gait_swing_drag_penalty,
        weight=-0.8,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": WHEEL_CONTACT_CFG,
            "gait_period": 0.8,
            "force_scale": 20.0,
        },
    )
    gait_both_legs_participation = RewTerm(
        func=gait_both_legs_participation_reward,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "asset_cfg": WHEEL_BODY_CFG,
            "sensor_cfg": WHEEL_CONTACT_CFG,
            "left_sensor_cfg": LEFT_WHEEL_SCANNER_CFG,
            "right_sensor_cfg": RIGHT_WHEEL_SCANNER_CFG,
            "gait_period": 0.8,
            "clearance_height": 0.12,
            "wheel_radius": 0.10,
        },
    )
    gait_lateral_drift = RewTerm(
        func=gait_lateral_drift_penalty,
        weight=-0.3,
        params={"command_name": "base_velocity"},
    )
    stance_wheel_lateral_slip = RewTerm(
        func=stance_wheel_lateral_slip_penalty,
        weight=-0.25,
        params={
            "command_name": "base_velocity",
            "asset_cfg": WHEEL_BODY_CFG,
            "sensor_cfg": WHEEL_CONTACT_CFG,
            "gait_period": 0.8,
            "max_penalty": 4.0,
        },
    )

    wheel_rolling = RewTerm(
        func=wheel_rolling_consistency,
        weight=0.20,
        params={"command_name": "base_velocity", "wheel_radius": 0.1, "asset_cfg": WHEEL_JOINT_CFG, "std": 0.35},
    )
    wheel_stop = RewTerm(
        func=wheel_not_stop_penalty,
        weight=-0.10,
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
class WheellegTerrianEnvCfg(WheellegEnvCfg):
    """WheelLeg locomotion environment over generated terrain."""

    scene: WheellegTerrianSceneCfg = WheellegTerrianSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: TerrianObservationsCfg = TerrianObservationsCfg()
    actions: TerrianActionsCfg = TerrianActionsCfg()
    commands: TerrianCommandsCfg = TerrianCommandsCfg()
    rewards: TerrianRewardsCfg = TerrianRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: TerrianEventCfg = TerrianEventCfg()
    curriculum: TerrianCurriculumCfg = TerrianCurriculumCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 20.0
        self.scene.contact_sensor.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.left_wheel_scanner.update_period = self.decimation * self.sim.dt
        self.scene.right_wheel_scanner.update_period = self.decimation * self.sim.dt
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # Pull the recording camera back and up so train videos show the
        # terrain strip and robot motion instead of a tight close-up.
        self.viewer.eye = (24.0, -38.0, 18.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)

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
