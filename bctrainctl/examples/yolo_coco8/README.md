# YOLO COCO8 示例

这个示例用于验证 `bctrainctl` 在阿里云 PAI DLC 上运行 YOLO 训练任务的完整链路。当前版本已经做过真实云端联调，适合用来验证代码打包、任务提交、日志查看和 OSS 输出回写。

## 这个示例会做什么

- 打包并上传 `examples/yolo_coco8/` 目录
- 在容器里安装 `ultralytics`
- 运行 `coco8.yaml` 数据集上的 YOLO 训练
- 把训练输出写到 `/workspace/output`
- 通过 OSS 挂载将输出保存到 `oss://oss-pai-wulanchabu-stark/bctrainctl/output/yolo-coco8-oss`

## 文件说明

- [job.yaml](job.yaml)：DLC 任务声明，定义镜像、quota、资源规格、代码打包目录和 OSS 输出挂载
- [train.sh](train.sh)：容器启动后的实际执行脚本，负责安装依赖、运行训练并打印输出文件列表

## 当前配置要点

- `spec.image` 使用和 AutoDL 私有云一致的 PyTorch 镜像
- `spec.resource_id` 当前是 `quota-4090`
- `spec.resources` 当前是 `gpu=1 cpu=16 memory_gb=60`
- `spec.code.project_path: .` 会被解析为 `examples/yolo_coco8/`
- `spec.code.use_gitignore`：设为 `true` 可启用 `.gitignore` 规则过滤打包文件
- `spec.code.gitignore_include`：`.gitignore` 中匹配但仍需打包的路径白名单
- `spec.runtime.workdir` 是 `/workspace/code`
- `train.sh` 里会执行 `pip install --no-cache-dir ultralytics`
- YOLO 输出目录固定写到 `/workspace/output`

## 前置条件

- 镜像内的 Python 环境可以正常安装 `ultralytics`
- 如果任务需要联网下载依赖或模型，容器网络需要可访问对应源站
- 目标 OSS 路径有写权限
- 如果环境需要代理，请确认 [train.sh](train.sh) 里的代理配置可用，或者直接改成你自己的代理地址

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
bctrainctl submit -f examples/yolo_coco8/job.yaml --dry-run
```

4. 真正提交任务：

```bash
bctrainctl submit -f examples/yolo_coco8/job.yaml
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
- `spec.image`：改成你自己的训练镜像
- `spec.resource_id`：改成当前 workspace 已绑定的 quota
- `spec.resources`：按任务需要调整 GPU、CPU 和内存
- `spec.storage.mounts[0].source`：改成你自己的输出目录，避免多人共用一个前缀

## 排障说明

- 如果 `pip install ultralytics` 失败，优先检查容器网络、镜像内证书和 Python 依赖环境
- 如果环境必须走代理，建议先确认脚本里的代理地址可用；必要时把代理导出提前到安装依赖之前
- 如果任务输出没有写回 OSS，先确认训练脚本是否仍然把 `project=/workspace/output`
- 如果只是想先验证提交和日志链路，可以临时移除 `storage.mounts`，排除挂载问题

## 联调结论

- 这套示例已经做过真实云端联调
- OSS 输出挂载链路已经验证
- 代码包当前走 OSS 签名 URL 下载，不再依赖 DataSource/PVC 挂载
- 直接在运行时安装 `ultralytics` 会改动镜像里的 Python 依赖栈
- 这个示例更适合作为“提交链路、日志链路、OSS 输出链路”的最小验证

如果后面要长期稳定跑 GPU 训练，更稳妥的做法是使用预装兼容 `torch/torchvision/ultralytics` 的镜像。
