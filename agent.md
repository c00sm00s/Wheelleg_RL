# Project Agent Notes

This file is a short project-level memory aid for future agents working in this
repository.

## Environment

- Project root: `/home/chenjunjie/Wheelleg/Wheelleg`
- Isaac Lab environment uses uv.
- Activate the correct environment before running project or Isaac Lab commands:

```bash
source /home/chenjunjie/env_isaaclab/bin/activate
```

- Do not assume `/home/chenjunjie/IsaacLab/env_isaaclab` is the active runtime
  environment. The user-provided environment is `/home/chenjunjie/env_isaaclab`.
- Prefer running Python/Isaac commands after activation, or through the matching
  uv environment if available.

## Important Paths

- Main task package:
  `source/Wheelleg/Wheelleg/tasks/manager_based/wheelleg`
- Base environment config:
  `source/Wheelleg/Wheelleg/tasks/manager_based/wheelleg/wheelleg_env_cfg.py`
- Terrain environment config:
  `source/Wheelleg/Wheelleg/tasks/manager_based/wheelleg/Wheelleg_Terrian_env_cfg.py`
- Jump environment config:
  `source/Wheelleg/Wheelleg/tasks/manager_based/wheelleg/Wheelleg_Jump_env_cfg.py`
- Custom MDP action terms:
  `source/Wheelleg/Wheelleg/tasks/manager_based/wheelleg/mdp/actions.py`
- RSL-RL runner config:
  `source/Wheelleg/Wheelleg/tasks/manager_based/wheelleg/agents/rsl_rl_ppo_cfg.py`

## Registered Gym Environments

Defined in:
`source/Wheelleg/Wheelleg/tasks/manager_based/wheelleg/__init__.py`

- `Wheelleg-Template-v0`
- `Wheelleg-AlternatingWalk-v0`
- `Wheelleg-Jump-v0`
- `Wheelleg-Terrian-v0`

Note: the project currently keeps the spelling `Terrian` in file names and env
IDs. Do not silently rename it to `Terrain`.

## Action Space Reminder

The intended policy action space is 6 continuous joint-position targets, not
joint efforts:

```text
[0] L_hip_front_joint
[1] L_hip_rear_joint
[2] R_hip_front_joint
[3] R_hip_rear_joint
[4] L_wheel_joint
[5] R_wheel_joint
```

Current normalized action mapping:

```text
hip target   = action * 0.82905 - 0.21815   # approx [-1.0472, 0.6109] rad
wheel target = action * pi                  # [-pi, pi] rad
```

RSL-RL clips raw actions to `[-1, 1]` via:

```python
clip_actions = 1.0
```

in `agents/rsl_rl_ppo_cfg.py`.

## Position-Control Notes

- `JointPositionActionCfg` is used for the base, jump, and terrain tasks.
- Position targets need non-zero actuator stiffness/damping to track. Do not set
  controlled hip/wheel actuator stiffness back to zero unless intentionally
  returning to effort control.
- Terrain lift/block rewards do not modify wheel action targets. Stage2/stage3
  no longer use any custom action that fixes or overrides wheel targets.
- Be careful with wheel position targets: continuous wheel joints controlled as
  absolute position targets can look locked at reset because zero action maps to
  a fixed wheel angle. If sustained rolling is required, prefer a hybrid action
  design: hip joints as position targets and wheel joints as velocity/effort or
  incremental position targets.
- Knee joints are intended to stay passive/damped in the jump and terrain
  setups.

## Terrain Reward Notes

- `Wheelleg-Terrian-v0` keeps global tracking rewards:
  `track_lin_vel_xy_exp` weight `3.0` and `track_ang_vel_z_exp` weight `1.2`.
- Stage2 and stage3 add extra tracking bonuses:
  `stage2plus_track_lin_vel_xy_exp` weight `2.0` and
  `stage2plus_track_ang_vel_z_exp` weight `0.8`.
- Therefore stage2/stage3 effective tracking weights are `5.0` for XY velocity
  and `2.0` for yaw rate.

## Terrain Reset Notes

- Stage2 uses raised inverted pyramid stairs. Keep reset near the low center
  platform; current stage2 reset range is `x/y=(-0.35, 0.35)`.
- Stage2 stair geometry is intentionally easier than before:
  `platform_width=6.0`, `step_width=0.90`, `step_height_range=(0.015, 0.045)`.
- Stage3 is a +X straight stair course. Current `bottom_platform_width=4.0`,
  so the first riser is at relative `x=2.0`.
- Stage3 reset is on the rear half of the bottom platform:
  `x=(-1.40, -0.80)`, `y=(-0.25, 0.25)`, `yaw=(-0.1, 0.1)`.
  This gives roughly `2.8 m` or more of flat approach before the first riser.

## Working Tree Caution

This repo may have existing uncommitted user changes. Before editing, check:

```bash
git status --short
```

Do not revert unrelated user changes.
