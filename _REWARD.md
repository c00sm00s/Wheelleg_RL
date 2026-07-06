# WheelLeg 奖励函数说明

本文记录 `Wheelleg-Terrian-v0` 当前启用的 reward 项、输入参数、权重和引导目标。

配置来源：
`source/Wheelleg/Wheelleg/tasks/manager_based/wheelleg/Wheelleg_Terrian_env_cfg.py`

自定义 reward 函数：
`source/Wheelleg/Wheelleg/tasks/manager_based/wheelleg/mdp/rewards.py`

Isaac Lab 内置 reward 函数通过 `mdp` 访问，因为 `mdp/__init__.py` 会导入
`isaaclab.envs.mdp`。

## 如何阅读

每个 reward 项对总奖励的贡献为：

```text
最终贡献 = weight * reward_function(env, **params)
```

- `env` 是隐式输入，不会在配置里显式写出。
- 正权重表示鼓励某种行为。
- 负权重表示惩罚某种行为。
- `SceneEntityCfg(...)` 用来选择机器人关节、机器人刚体或接触传感器刚体。

## 常用输入

- `command_name="base_velocity"`：速度指令，包括线速度和角速度。
- `ACTUATED_JOINT_CFG`：主动驱动关节，包括 4 个 hip 和 2 个 wheel。
- `HIP_JOINT_CFG`：4 个 hip 驱动关节。
- `KNEE_JOINT_CFG`：4 个被动 knee 连杆关节。
- `BASE_BODY_CFG`：`base_link`。
- `WHEEL_JOINT_CFG`：`L_wheel_joint`、`R_wheel_joint`。
- `WHEEL_BODY_CFG`：`L_wheel_link`、`R_wheel_link`。
- `WHEEL_CONTACT_CFG`：左右轮子的接触传感器 body。
- `LEG_LINK_CONTACT_CFG`：大腿/小腿接触 body，不包含轮子。
- `left_foot_scanner` / `right_foot_scanner`：足端局部 ray 网格，用于 privileged terrain observation 和 ray-based 轮底离地高度。
- `STAGE2_INDEX = 2`：金字塔/上楼梯过渡阶段。
- `STAGE3_INDEX = 3`：单方向直楼梯阶段。
- `STAGE3_STAIRS_CFG`：stage3 直楼梯几何参数。

## 当前启用的 Reward

| 名称 | 函数 | 主要输入 | 权重 | 引导目标 |
| --- | --- | --- | ---: | --- |
| `track_lin_vel_xy_exp` | `mdp.track_lin_vel_xy_exp` | `command_name="base_velocity"`, `std=0.45` | `3.0` | 让机体系 XY 线速度跟踪速度指令，是主要前进/横移追踪奖励。 |
| `track_ang_vel_z_exp` | `mdp.track_ang_vel_z_exp` | `command_name="base_velocity"`, `std=0.45` | `1.2` | 让 yaw 角速度跟踪转向指令。 |
| `stage2plus_track_lin_vel_xy_exp` | `stage_track_lin_vel_xy_exp` | `command_name="base_velocity"`, `min_stage=STAGE2_INDEX`, `std=0.45` | `2.0` | stage2/stage3 额外增强 XY 速度追踪，抑制只抬腿、不按 command 前进。 |
| `stage2plus_track_ang_vel_z_exp` | `stage_track_ang_vel_z_exp` | `command_name="base_velocity"`, `min_stage=STAGE2_INDEX`, `std=0.45` | `0.8` | stage2/stage3 额外增强 yaw 角速度追踪，避免爬障时忽略转向/朝向控制。 |
| `alive` | `mdp.is_alive` | 无 | `0.5` | 每个存活 step 给奖励，鼓励不摔倒、不提前终止。 |
| `terminating` | `mdp.is_terminated` | 无 | `-2.0` | 对失败终止进行惩罚。 |
| `flat_orientation_l2` | `mdp.flat_orientation_l2` | 默认 robot asset | `-2.0` | 惩罚 roll/pitch 倾斜，鼓励机身保持水平。 |
| `lin_vel_z_l2` | `mdp.lin_vel_z_l2` | 默认 robot asset | `-0.50` | 惩罚 base 竖直速度，减少上下弹跳，同时保留爬台阶所需高度变化。 |
| `ang_vel_xy_l2` | `mdp.ang_vel_xy_l2` | 默认 robot asset | `-0.08` | 惩罚 roll/pitch 角速度，减少机身摇晃和翻滚。 |
| `base_height_l2` | `safe_base_height_l2` | `target_height=0.50`, `sensor_cfg=height_scanner` | `-0.50` | 让 base 高度接近地形相对目标高度，并对 ray hit 做 NaN/Inf 防护。 |
| `action_rate_l2` | `mdp.action_rate_l2` | 当前 action 和上一帧 action | `-0.02` | 惩罚动作变化过快，让目标关节角更平滑。 |
| `joint_torque_l2` | `mdp.joint_torques_l2` | `asset_cfg=ACTUATED_JOINT_CFG` | `-1.0e-5` | 惩罚过大关节力矩，降低电机负担。 |
| `joint_power_l1` | `mdp.joint_power_l1` | `asset_cfg=ACTUATED_JOINT_CFG` | `-2.0e-6` | 惩罚机械功率 `abs(torque * joint_velocity)`，减少高功耗动作。 |
| `joint_acc_l2` | `mdp.joint_acc_l2` | `asset_cfg=ACTUATED_JOINT_CFG` | `-2.5e-7` | 惩罚关节加速度，提高关节运动平滑性。 |
| `base_lin_acc_l2` | `mdp.body_lin_acc_l2` | `asset_cfg=BASE_BODY_CFG` | `-2.0e-4` | 惩罚 base 线加速度，减少机身冲击。 |
| `joint_velocity_limits` | `joint_velocity_limit_penalty` | `asset_cfg=ACTUATED_JOINT_CFG`, `soft_ratio=0.90` | `-0.08` | 惩罚接近或超过软速度限制的关节速度，保护执行器。 |
| `leg_motion` | `leg_joint_motion_reward` | `command_name="base_velocity"`, `asset_cfg=HIP_JOINT_CFG`, `min_stage=1` | `0.05` | 在粗糙/楼梯阶段，轻微鼓励有用的 hip 运动，但要求速度追踪和姿态已经较好。 |
| `leg_limit_margin` | `leg_joint_limit_margin_penalty` | `asset_cfg=HIP_JOINT_CFG`, `margin=0.80` | `-0.12` | 惩罚 hip 长期贴近位置极限，避免学成刚性撑杆。 |
| `five_bar_singularity` | `five_bar_singularity_penalty` | `asset_cfg=KNEE_JOINT_CFG`, `margin=0.75`, `exponent=4.0`, `max_penalty=4.0` | `-0.20` | 惩罚被动 knee 连杆接近行程极限，避免五连杆接近折叠/打直的奇异姿态。 |
| `wheel_clearance_difference` | `wheel_clearance_difference_penalty` | `asset_cfg=WHEEL_BODY_CFG`, `left/right_foot_scanner`, `min_stage=STAGE2_INDEX`, `deadband=0.16`, `std=0.05` | `-0.25` | stage2 及以上惩罚左右轮底离地高度差过大，抑制一条腿极端拉伸、另一条腿极端压缩。 |
| `gait_alternating_contact` | `gait_alternating_contact_reward` | `command_name="base_velocity"`, `sensor_cfg=WHEEL_CONTACT_CFG`, `gait_period=0.75`, `double_support_width=0.14` | `0.25` | 轻微引导左右轮腿交替支撑/摆动，类似 trot，但权重较小，允许爬障时打破节律。 |
| `gait_swing_drag` | `gait_swing_drag_penalty` | `command_name="base_velocity"`, `sensor_cfg=WHEEL_CONTACT_CFG`, `gait_period=0.75`, `force_scale=20.0` | `-0.20` | 惩罚摆动侧轮子仍然受力，减少拖地。 |
| `gait_lateral_drift` | `gait_lateral_drift_penalty` | `command_name="base_velocity"` | `-0.3` | 惩罚运动时机体系横向漂移，让运动更干净。 |
| `undesired_leg_contacts` | `mdp.undesired_contacts_count` | `sensor_cfg=LEG_LINK_CONTACT_CFG`, `threshold=1.0` | `-1.5` | 惩罚大腿/小腿与地形接触，正常情况下只允许轮子主动接触地形。 |
| `history_blocked_swing_clearance` | `history_blocked_swing_clearance_reward` | `WHEEL_BODY_CFG`, `WHEEL_CONTACT_CFG`, `LEG_LINK_CONTACT_CFG`, `left/right_foot_scanner`, `lift_height=0.10`, `wheel_radius=0.0625`, `min_blocked_stage=STAGE3_INDEX` | `1.25` | stage3 中，当历史速度误差、水平接触力或轮子空转判定 blocked 后，奖励 blocked 轮子的轮底高于 ray 观测地形。 |
| `stage2_blocked_asymmetric_lift` | `stage_blocked_asymmetric_wheel_lift_reward` | `stage=STAGE2_INDEX`, `left/right_foot_scanner`, `height_margin=0.035`, `min_base_height=0.42`, `min_forward_speed=0.08`, `max_lift_height=0.16`, `max_wheel_clearance_difference=0.20`, `gait_period=0.75`, `swing_fraction=0.35` | `0.75` | stage2 中，blocked 后奖励单侧非对称抬轮，但必须保持机身高度、沿 command 有前进速度，并避免过度抬腿或左右腿高度差过大。 |
| `history_blocked_lift_velocity` | `history_blocked_lift_velocity_reward` | `target_lift_speed=0.35`, `min_blocked_stage=2` | `0.35` | stage2 及以上，blocked 后奖励轮端相对 base 向上运动，鼓励主动抬腿。 |
| `history_blocked_drag` | `history_blocked_drag_penalty` | `min_blocked_stage=2` | `-1.0` | blocked 后惩罚摆动侧继续承重或拖地，减少硬蹭障碍。 |
| `stage3_stair_progress` | `stage3_stair_progress_reward` | `stage=STAGE3_INDEX`, 直楼梯几何, `asset_cfg=WHEEL_BODY_CFG`, `wheel_radius=0.0625`, `height_margin=0.04` | `2.0` | stage3 中，只有左右轮腿都确认到达新台阶时才奖励进度。 |
| `stage3_split_stair_lagging_lift` | `stage3_split_stair_lagging_lift_reward` | `stage=STAGE3_INDEX`, `left/right_foot_scanner`, `lift_height=0.125`, `std=0.05` | `1.0` | 如果一侧已经上到更高一级而另一侧落后，奖励落后侧轮底高于 ray 观测地形。 |
| `stage3_heading_alignment` | `stage3_heading_alignment_reward` | `stage=STAGE3_INDEX`, `target_heading=0.0`, `std=0.35` | `0.5` | stage3 中鼓励机器人朝向对准上楼方向，也就是世界系 +X。 |
| `wheel_rolling` | `wheel_rolling_consistency` | `command_name="base_velocity"`, `wheel_radius=0.0625`, `asset_cfg=WHEEL_JOINT_CFG`, `std=0.45` | `0.35` | 鼓励轮子角速度对应的滚动速度与前进速度指令一致。 |
| `wheel_stop` | `wheel_not_stop_penalty` | `command_name="base_velocity"`, `asset_cfg=WHEEL_JOINT_CFG` | `-0.30` | 有前进指令但两个轮子几乎停止时惩罚。 |

## Blocked 判定

当前所有 blocked 抬腿相关 reward 共享 `_history_blocked_wheel_gate`。这个 gate 只影响 reward，不再修改 action 里的轮子目标角。

轮子被认为 blocked 的核心逻辑是：

```text
moving
+ wheel_contact
+ (
    wheel_velocity_error_large
    or horizontal_contact_force_large
    or wheel_spin_without_forward_progress
  )
```

其中：

- `moving`：速度指令大小大于 `min_command_speed=0.1`。
- `wheel_contact`：轮子接触力超过 `contact_threshold=1.0`。
- `wheel_velocity_error_large`：最近 `history_length=5` 帧中，轮端 XY 速度跟不上 `command_xy`，阈值 `wheel_velocity_error_threshold=0.25`。
- `horizontal_contact_force_large`：轮子接触力有明显水平分量，用于识别撞到台阶竖直面：
  ```text
  ||F_xy|| > horizontal_force_threshold=1.0
  ||F_xy|| / |F_z| > horizontal_force_ratio_threshold=0.4
  ```
- `wheel_spin_without_forward_progress`：轮子转得很快，但 base 或 wheel link 沿指令方向前进很慢：
  ```text
  abs(wheel_joint_vel) * wheel_radius > wheel_spin_speed_threshold=0.5
  forward_speed < wheel_spin_forward_speed_threshold=0.12
  ```
- `LEG_LINK_CONTACT_CFG` 中的大腿/小腿接触也会直接作为 blocked 辅助信号。
- blocked latch 会保持 `blocked_memory_steps=30` 步，避免刚开始抬腿、接触消失时 reward 立刻断掉。
- stage2/stage3 不再固定或覆盖轮子目标角；blocked 信号只用于奖励抬腿、惩罚拖地，不会覆盖策略输出的轮子目标角。

## Stage2 防止缩腿/一伸一缩

stage2 的 `stage2_blocked_asymmetric_lift` 现在不是“只要轮底离地就给奖励”，还额外乘上这些 gate：

- `min_base_height=0.42`：base 太低时抬腿奖励衰减，避免缩成一团。
- `min_forward_speed=0.08`：沿 command 方向几乎不前进时抬腿奖励衰减。
- `max_lift_height=0.16`：抬得过高也衰减，避免靠极端收腿刷 reward。
- `max_wheel_clearance_difference=0.20`：左右轮底高度差过大时衰减。

另外新增 `wheel_clearance_difference` 惩罚项，左右轮底离地高度差超过 `deadband=0.16` 后开始扣分，用来抑制“一条腿拉伸、一条腿压缩”的姿态。

## 抬腿高度定义

现在 stage2/stage3 的抬腿高度统一按 ray 特权信息定义：

```text
lift_height = wheel_bottom_z - ray_ground_z
wheel_bottom_z = wheel_link_z - wheel_radius
wheel_radius = 0.0625
```

`ray_ground_z` 来自对应的 `left_foot_scanner` / `right_foot_scanner`。实现上取足端 ray 网格中 XY 距离轮心最近的有效 ray hit 作为地面高度。

## 当前禁用项

| 名称 | 当前值 | 说明 |
| --- | --- | --- |
| `gait_swing_clearance` | `None` | 禁用，因为 blocked/stair 专用抬腿 reward 已经提供地形相关 clearance。 |
| `gait_both_legs_participation` | `None` | 禁用，避免爬障时强制固定步态。 |
| `stance_wheel_lateral_slip` | `None` | 禁用，目前使用 rolling consistency 和 lateral drift 惩罚替代。 |

## 按目标分组

### 速度追踪

- `track_lin_vel_xy_exp`
- `track_ang_vel_z_exp`
- `stage2plus_track_lin_vel_xy_exp`
- `stage2plus_track_ang_vel_z_exp`
- `wheel_rolling`
- `stage3_heading_alignment`

这些项负责让机器人跟踪线速度、角速度、轮子滚动速度和 stage3 上楼方向。其中 stage2/stage3 的线速度 tracking 总权重为 `3.0 + 2.0 = 5.0`，yaw tracking 总权重为 `1.2 + 0.8 = 2.0`。

### 能耗和执行器保护

- `joint_torque_l2`
- `joint_power_l1`
- `joint_velocity_limits`
- `action_rate_l2`

这些项减少力矩、功率、动作突变和接近速度极限的行为。

### 平稳性和机身安全

- `flat_orientation_l2`
- `lin_vel_z_l2`
- `ang_vel_xy_l2`
- `base_height_l2`
- `joint_acc_l2`
- `base_lin_acc_l2`
- `leg_limit_margin`
- `five_bar_singularity`
- `wheel_clearance_difference`

这些项让机身更稳定，减少弹跳、摇晃和危险连杆姿态。

### 接触和步态引导

- `gait_alternating_contact`
- `gait_swing_drag`
- `gait_lateral_drift`
- `undesired_leg_contacts`

这些项轻微引导交替接触，并惩罚拖地、横向漂移和腿部非预期碰撞。

### 地形和楼梯通过

- `leg_motion`
- `history_blocked_swing_clearance`
- `stage2_blocked_asymmetric_lift`
- `history_blocked_lift_velocity`
- `history_blocked_drag`
- `stage3_stair_progress`
- `stage3_split_stair_lagging_lift`

这些项负责粗糙地形/楼梯阶段的 blocked 检测、主动抬腿、避免拖拽和确认台阶进度。

## 基础任务 Reward

`TerrianRewardsCfg` 继承自 `wheelleg_env_cfg.py` 中的 `RewardsCfg`，但覆盖了大部分基础 locomotion reward，并添加了地形相关 reward。

基础任务 `Wheelleg-Template-v0` 使用较简单的 reward：

| 名称 | 函数 | 权重 | 引导目标 |
| --- | --- | ---: | --- |
| `track_lin_vel_xy_exp` | `mdp.track_lin_vel_xy_exp` | `1.5` | 跟踪 XY 速度指令。 |
| `track_ang_vel_z_exp` | `mdp.track_ang_vel_z_exp` | `1.0` | 跟踪 yaw 角速度指令。 |
| `alive` | `mdp.is_alive` | `1.0` | 保持存活。 |
| `terminating` | `mdp.is_terminated` | `-2.0` | 惩罚失败终止。 |
| `flat_orientation_l2` | `mdp.flat_orientation_l2` | `-2.0` | 保持机身水平。 |
| `lin_vel_z_l2` | `mdp.lin_vel_z_l2` | `-2.0` | 减少竖直弹跳。 |
| `ang_vel_xy_l2` | `mdp.ang_vel_xy_l2` | `-0.05` | 减少 roll/pitch 角速度。 |
| `action_rate_l2` | `mdp.action_rate_l2` | `-0.01` | 平滑动作。 |
| `joint_torque_l2` | `mdp.joint_torques_l2` | `-1.0e-5` | 减少力矩使用。 |
