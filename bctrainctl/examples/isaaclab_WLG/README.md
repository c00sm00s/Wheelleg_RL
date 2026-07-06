# IsaacLab 示例

这个示例用于验证 `bctrainctl` 在阿里云 PAI DLC 上运行 IsaacLab 镜像的完整链路。当前版本已经完成真实云端联调，训练可以正常完成，输出也可以正常写回 OSS。

## 这个示例会做什么

- 打包并上传 `examples/isaaclab/` 目录
- 在容器里激活镜像内预装的 `isaaclab` conda 环境
- 执行 IsaacLab 官方 `Isaac-Cartpole-v0` 的 RSL-RL 训练
- 把训练输出写到 `/workspace/output`
- 通过 OSS 挂载将输出保存到 `oss://oss-pai-wulanchabu-stark/bctrainctl/output/isaaclab_test`

## 文件说明

- [job.yaml](job.yaml)：DLC 任务声明，定义镜像、quota、资源规格、代码打包目录和 OSS 输出挂载
- [train.sh](train.sh)：容器启动后的实际执行脚本，负责激活环境、运行训练并打印输出文件列表

## 当前配置要点

- `spec.image` 使用预装 IsaacLab 的镜像
- `spec.resource_id` 当前是 `quota-4090`
- `spec.resources` 当前是 `gpu=1 cpu=12 memory_gb=60`
- `spec.code.project_path: .` 会被解析为 `examples/isaaclab/`
- `spec.code.use_gitignore`：设为 `true` 可启用 `.gitignore` 规则过滤打包文件
- `spec.code.gitignore_include`：`.gitignore` 中匹配但仍需打包的路径白名单
- `spec.runtime.workdir` 是 `/workspace/code`
- `train.sh` 会先 `cd /workspace/output`，再启动训练，确保输出直接落到挂载目录

## 前置条件
- 所选 quota 和镜像驱动栈可以正常运行 IsaacLab
- 目标 OSS 路径有写权限
- 如果环境不需要代理，或者 `10.0.0.152:7890` 不可用，请先删除 [train.sh](train.sh) 里的代理配置

## 运行步骤

1. 在仓库根目录安装并初始化 CLI：

```bash
python -m pip install -e .
bctrainctl init
```

2. 检查当前配置：

```bash
bctrainctl config show
```

3. 先做一次 dry-run，确认镜像、quota 和挂载参数：

```bash
bctrainctl submit -f examples/isaaclab/job.yaml --dry-run
```

4. 真正提交任务：

```bash
bctrainctl submit -f examples/isaaclab/job.yaml
```

5. 跟日志：

```bash
bctrainctl logs <job_id> --follow --lines 200
```

6. 查看任务详情：

```bash
bctrainctl show <job_id>
```

7. 如需停止或删除：

```bash
bctrainctl stop <job_id>
bctrainctl delete <job_id>
```

## 提交前建议改的字段

- `metadata.project`：改成你的项目名
- `spec.image`：改成你自己的 IsaacLab 镜像
- `spec.resource_id`：改成当前 workspace 已绑定的 quota
- `spec.resources`：按任务需要调整 GPU、CPU 和内存
- `spec.storage.mounts[0].source`：改成你自己的输出目录，避免多人共用一个前缀

## 排障说明
- 如果 `/root/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py` 不存在，说明镜像不是当前脚本假设的 IsaacLab 镜像
- 如果只是想先验证提交和日志链路，可以临时移除 `storage.mounts`，排除挂载问题
- 如果任务输出没有写回 OSS，先确认训练脚本是否仍然把工作目录切到了 `/workspace/output`

