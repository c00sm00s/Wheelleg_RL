# bctrainctl

`bctrainctl` 是 BrainCo 的本地训练任务 CLI，用来把训练任务提交到阿里云 PAI DLC，并在本地维护配置、任务索引、代码包和调试信息。

## 安装

提示：isaacsim 5.1.0 把所有依赖都精确固定了，为避免潜在的isaacsim环境冲突，推荐在一个独立的python 3.11以上虚拟环境中安装：

使用源码安装：

```bash
# 克隆bctrainctl仓库
git clone git@gitlab.brainco.cn:hpc_infra/bctrainctl.git
# 使用pip命令安装bctrainctl
python -m pip install -e .
```

使用release中提供的whl安装：

```bash
# 手动下载版本发布中的最新whl文件
# 使用以下命令安装whl
python -m pip install bctrainctl-*.whl
```

安装后主命令是：

```bash
bctrainctl --help
```

如果 shell 找不到命令，也可以直接运行：

```bash
python -m bctrainctl --help
```

## 初始化

```bash
bctrainctl init
```

初始化会引导填写阿里云凭证、region、workspace、OSS bucket、默认 quota/镜像/资源规格等。详见 [配置文档](docs/configuration.md)。

## 命令概览

| 命令 | 说明 |
| --- | --- |
| `bctrainctl tui` | 打开交互式控制台（新建任务 / 推送 ACR 镜像 / 编辑配置） |
| `bctrainctl init` | 初始化本地配置并验证云端连通性（默认询问是否用 TUI 配置） |
| `bctrainctl config` | 直接打印当前配置（脱敏）并给出 Usage 提示 |
| `bctrainctl config show` | 查看当前配置（脱敏） |
| `bctrainctl config set <key> <value>` | 修改单个配置字段 |
| `bctrainctl config edit` | 打开 TUI 配置界面交互编辑 |
| `bctrainctl submit -f <job.yaml>` | 打包代码、上传 OSS、提交 DLC 训练任务 |
| `bctrainctl submit` | 不带 `-f` 时打开 TUI 交互填写并提交任务 |
| `bctrainctl acr push` | 选择本地 Docker 镜像并推送到阿里云 ACR |
| `bctrainctl list` | 列出本地已跟踪任务（自动同步云端状态） |
| `bctrainctl list --remote` | 列出 workspace 全部任务 |
| `bctrainctl show <job_id>` | 查看任务详情 |
| `bctrainctl logs <job_id>` | 查看任务日志（默认轮询，`--no-follow` 只拉一次） |
| `bctrainctl stop <job_id>` | 停止任务 |
| `bctrainctl delete <job_id>` | 删除任务（同时清理 OSS 代码包） |
| `bctrainctl sync` | 批量同步本地任务状态 |

全局参数：`--debug` 打印 SDK 请求参数、响应和堆栈。

## 交互式 TUI

`bctrainctl tui` 打开一个基于 [textual](https://textual.textualize.io/) 的交互控制台，主菜单按使用频率排列：**①新建任务 → ②推送 ACR 镜像 → ③编辑配置**。不熟悉 YAML 字段时，用 TUI 填表是最省心的方式。

### 启动入口

| 入口 | 进入的界面 |
| --- | --- |
| `bctrainctl tui` | 主菜单（任务 / ACR / 配置三选一） |
| `bctrainctl submit`（不带 `-f`） | 直接进入「新建任务」 |
| `bctrainctl acr push --tui` | 直接进入「推送 ACR 镜像」 |
| `bctrainctl config edit` | 直接进入「编辑配置」 |
| `bctrainctl init`（默认） | 询问「是否用 TUI 配置」，确认后进入「编辑配置」 |

### 三大功能

- **新建任务**：表单填写 job 参数，确认后写出临时 `job.yaml`，再走与 `submit` 完全一致的打包/上传/提交流程。
  - **加载**：下拉框加载已保存模板；或「Load file」**从当前运行目录直接读取项目里的 YAML**（默认填 `job.yaml`，可改路径），方便「读现有模板 → 改 → 另存」。
  - **保存**：保存框填**模板名**则存到 `~/.bctrainctl/templates/`；填**路径**（如 `./my-job.yaml`）则存到该路径。
- **推送 ACR 镜像**：选择本地 Docker 镜像、填写 ACR 目标与凭证，确认后**退出 TUI 在普通终端执行** docker login/tag/push（这样 docker 输出能正常显示）。
- **编辑配置**：查看/修改阿里云 access/secret key、wandb / swanlab key、ACR 凭证与各项默认值。首次配置（本地还没有 config）时会**预填 init 的默认参数**（region、workspace、OSS bucket/prefix、quota、镜像、ACR registry 等）。

### 界面操作

- **↑/↓ 方向键**在字段间切换焦点（也支持 Tab / Shift+Tab）
- 多行输入框（Exclude / env / extra_uploads 等）内 ↑/↓ 为移动光标
- `Ctrl+S` 提交/保存，`Esc` 返回上一层，主菜单按 `q` 退出

## 推送 Docker 镜像到 ACR

两种方式：

```bash
# 1) 交互式 CLI：交互选择本地镜像，使用 config 里的 ACR 凭证登录并推送
bctrainctl acr push
bctrainctl acr push --image foo:1.0 --name foo --tag 1.0

# 2) TUI：表单填写后执行
bctrainctl acr push --tui
```

registry / namespace / username / password 默认取自 `bctrainctl config`，可用命令行参数（或 TUI 表单）覆盖；macOS 等无需 sudo 的环境默认 `--no-sudo`，PAI/Linux 上可加 `--sudo`。

各命令的详细执行流程见 [命令详解](docs/commands.md)。

## 示例项目

- **YOLO COCO8** — [README](examples/yolo_coco8/README.md) | [job.yaml](examples/yolo_coco8/job.yaml)
- **IsaacLab** — [README](examples/isaaclab/README.md) | [job.yaml](examples/isaaclab/job.yaml)

## 更多文档

- [配置文档](docs/configuration.md) — 本地目录结构、凭证来源、初始化详情、代码打包配置
- [命令详解](docs/commands.md) — 各命令的完整执行流程
- [阿里云权限](docs/aliyun-permissions.md) — API 权限点、推荐 RAM 策略、官方文档链接
- [更新日志](docs/changelog.md) — 各版本的更新内容
