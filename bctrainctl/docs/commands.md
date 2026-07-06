# 命令详解

## `bctrainctl config`

不带子命令时直接打印当前配置（脱敏），并在末尾给出 `Usage: bctrainctl config [OPTIONS] COMMAND [ARGS]...` 提示。

子命令：

- `config show`：读取 `config.yaml`，用 Pydantic 校验后按脱敏方式打印（`access_key_secret`、`wandb_api_key`、`swanlab_api_key`、`acr_password` 都不会明文输出）
- `config set <key> <value>`：修改单个字段，写回 `config.yaml` 前会重新做整体校验
- `config edit`：打开 TUI 配置界面交互编辑

这些命令都不调用云端 API。

## `bctrainctl tui`

打开基于 textual 的交互控制台，主菜单提供：

1. 编辑配置：等价于 `config edit`，可改阿里云 / wandb / swanlab / ACR 凭证与默认值
2. 新建任务：表单填写 job 参数；点击「Submit」后写出一个临时 job.yaml，再走与 `submit` 完全一致的打包/上传/提交流程
   - 「Load file」可从**当前运行目录**读取任意 YAML（路径如 `./job.yaml`），方便直接读取项目里的模板再另存；下拉框可加载 `~/.bctrainctl/templates/` 下已保存的模板
   - 「Save」框填**模板名**则存到 `~/.bctrainctl/templates/`，填**路径**（含 `/` 或以 `.yaml`/`.yml` 结尾，如 `./my-job.yaml`）则存到该路径
   - 字段间可用 **↑/↓ 方向键**切换（多行输入框内 ↑/↓ 移动光标）
3. 推送 ACR 镜像：等价于 `acr push --tui`，表单选择本地镜像并填写 ACR 目标；点击「Push」后退出 TUI，在普通终端执行 docker login/tag/push（这样 docker 的输出能正常显示）

`bctrainctl submit` 不带 `-f` 时会直接进入「新建任务」界面。

## `bctrainctl acr push`

把本地 Docker 镜像推送到阿里云 ACR，是 `aliyun_acr_tools/acr_shell/acr_push.sh` 的 Python 实现。提供两种方式：

### 交互式 CLI（默认）

1. 检查本地 `docker`（以及 `--sudo` 时的 `sudo`）是否可用
2. 未指定 `--image` 时列出本地可打标签的镜像，交互选择一个
3. registry / namespace / username / password 默认取自 `bctrainctl config`，可用命令行参数覆盖；缺失时交互询问
4. 组装目标地址 `registry/namespace/name:tag` 并打印推送计划
5. 确认后执行 `docker login --password-stdin` → `docker tag` → `docker push`

### TUI（`acr push --tui` 或 `bctrainctl tui` 主菜单）

1. 表单界面：从本地镜像列表选择（可「Refresh」重新列出）或手动输入镜像 ref，填写 registry / namespace / 凭证 / 远端 name·tag、是否 sudo
2. 凭证等默认从 `bctrainctl config` 预填
3. 点击「Push」后退出 TUI，在普通终端打印推送计划并执行 login/tag/push（放到 TUI 外执行，docker 输出才能正常显示）

说明：

- macOS 等无需 sudo 的环境默认 `--no-sudo`，PAI/Linux 上可加 `--sudo`
- `--yes/-y` 可跳过 CLI 方式的确认

## `bctrainctl submit -f job.yaml`

这是核心链路。执行流程如下。

### 本地阶段

1. 读取 `job.yaml`
2. 用 Pydantic 校验 schema
3. 解析默认值，例如默认 quota、默认镜像、默认资源规格
4. 校验 `project_path` 是否存在
5. 校验 `entrypoint`、资源规格、挂载 target 去重
6. 从 `spec.code.project_path` 遍历本地项目
7. 应用默认排除规则和 `exclude` 规则（基于 `pathspec` 的 gitwildmatch，与 Git 同款语义，支持 `**`、目录规则、`!` 取反）
8. 如果 `use_gitignore: true`，额外读取项目根目录 `.gitignore` 中的规则（含 `!` 取反行）
9. 如果设置了 `gitignore_include`，匹配的路径即使被 `.gitignore` 命中也会保留打包
10. 打包成 `tar.gz`
11. 计算代码包 `sha256`
12. 将本地包保存到 `~/.bctrainctl/packages/`
13. 如果设置了 `spec.extra_uploads`，对每个 `local` 单独打成 `tar.gz`（目录打内容、单文件保留文件名）

### OSS 阶段

1. 生成 OSS object key，当前格式是：

```text
<oss_prefix>/code_packages/<job_name>-<timestamp>-<sha256>.tar.gz
```

2. 上传代码包到 OSS
3. 生成代码包下载签名 URL
4. 如果有 `spec.extra_uploads`，把每个额外包上传到 `<oss_prefix>/extra_uploads/...` 并各自生成下载签名 URL
5. 如果 `spec.storage.mounts` 里有 OSS 挂载，会额外往目标前缀写一个 `.bctrainctl_keep` 占位对象

### DLC 请求组装阶段

1. 调用 AIWorkspace `ListResources`
2. 把 `spec.resource_id` 里的 quota 名称解析成真正的工作空间绑定 `QuotaId`
3. 组装 DLC `CreateJob`
4. 当前固定创建 `PyTorchJob`
5. 当前固定单 worker：`job_specs=[Worker x 1]`
6. 资源规格来自 `spec.resources`
7. 如果配置了 `spec.priority`，会写入 DLC `CreateJob.Priority`
8. `storage.mounts` 会被转换为 DLC `DataSources`

### 容器内执行阶段

1. DLC 启动容器
2. 容器先通过签名 URL 把代码包下载到 `/tmp/<package>.tar.gz`
3. 将代码包解压到 `spec.runtime.workdir`
4. 如果有 `spec.extra_uploads`，依次把每个额外包下载并解压到对应的 `target` 目录
5. `cd` 到 `workdir`
6. 执行 `entrypoint`

### 本地索引阶段

1. 记录 `submission_id`
2. 记录 `job_id`
3. 记录 `oss_code_uri`
4. 记录 `package_hash`
5. 记录原始 spec 和请求快照
6. 写入 `~/.bctrainctl/jobs.db`

提交完成后会输出：

- Job Name
- DLC Job ID
- Status
- OSS Code URI
- Entrypoint
- Image
- DLC Quota
- Priority
- Resources
- Package Path
- Logs Command

## `bctrainctl list`

默认行为不是单纯读本地，而是：

1. 读取本地 `jobs.db`
2. 如果本地已有任务记录，调用 DLC `ListJobs`
3. 用云端状态刷新本地已跟踪任务
4. 再显示本地任务列表

因此默认 `list` 的语义是：

- 展示本地已跟踪任务
- 但状态会尽量先同步成云端最新值

## `bctrainctl list --remote`

执行流程：

1. 调用 DLC `ListJobs`
2. 分页拉取当前 workspace 的全部任务
3. 如果某个任务本地也有记录，则顺手更新本地状态
4. 渲染完整的 workspace 任务视图

说明：

- 云端有、本地没有的任务也会显示
- 这类任务的 `project` 会显示为 `-`

## `bctrainctl show <job_id>`

执行流程：

1. 调用 DLC `GetJob`
2. 获取该任务的云端最新状态、时间和原因字段
3. 如果本地有该任务记录，则顺手刷新本地状态和远端 payload
4. 用本地元数据补充 `project`、`OSS Code URI`、`Package Hash` 等字段
5. 输出合并后的最新详情

说明：

- `show` 现在默认以云端最新数据为准
- 如果该任务本地没有记录，但云端存在，仍然可以展示云端详情

## `bctrainctl logs <job_id>`

执行流程：

1. 调用 DLC `GetJob`
2. 从返回结果里拿 Pod 列表
3. 取第一个有 `PodId` 的可用 Pod
4. 调用 DLC `GetPodLogs`
5. 默认轮询刷新日志；如果带 `--no-follow`，则只抓一次

说明：

- 当前默认就是轮询刷新日志
- 如果只想抓一次，可以显式加 `--no-follow`
- 当前只读取第一个有 `PodId` 的可用 Pod 日志，不是 DLC 的平台限制，而是当前 CLI 的简化实现
- 如果任务还没起 Pod，会报错提示

## `bctrainctl stop <job_id>`

执行流程：

1. 调用 DLC `StopJob`
2. 如果本地有该任务记录，则把本地状态更新成 `STOPPED`
3. 输出停止结果

## `bctrainctl delete <job_id>`

执行流程：

1. 调用 DLC `DeleteJob`
2. 如果本地记录里有 `oss_code_uri`，则删除对应 OSS 代码包对象
3. 从本地 `jobs.db` 删除任务记录
4. 输出删除结果

说明：

- 删除的是提交时上传的代码包对象
- 不会删除训练输出目录
- 如果本地没有该任务记录，就拿不到 `oss_code_uri`，此时 OSS 删除会显示为 `SKIPPED`

## `bctrainctl sync`

执行流程：

1. 读取本地全部任务记录
2. 调用 DLC `ListJobs`
3. 用云端返回结果批量刷新本地状态
4. 输出本次同步数量
