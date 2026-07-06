# 更新日志

本项目遵循语义化版本。日期为本地时间。

## v0.2.0 - 2026-06-16

本版本新增交互式 TUI、ACR 镜像推送、额外文件上传，并重写了打包过滤引擎，同时修复了若干体验问题。

### 新增

- **交互式 TUI（`bctrainctl tui`）**：基于 textual 的控制台，主菜单含三项功能
  - **新建任务**：表单填写 job 参数后走与 `submit` 一致的提交流程；支持加载已保存模板，也可从当前运行目录直接读取项目里的 YAML（默认 `job.yaml`）；保存时可填模板名（存到 `~/.bctrainctl/templates/`）或保存路径（如 `./my-job.yaml`）
  - **推送 ACR 镜像**：表单选择本地镜像、填写目标，确认后退出 TUI 在终端执行 docker login/tag/push
  - **编辑配置**：查看/修改阿里云、wandb / swanlab、ACR 凭证；首次配置预填 init 默认参数
  - 入口：`bctrainctl tui` / `submit`（不带 `-f`）/ `acr push --tui` / `config edit` / `init`（默认询问）
  - 操作：↑/↓ 方向键切换字段（也支持 Tab），多行框内 ↑/↓ 移动光标，`Ctrl+S` 提交、`Esc` 返回、主菜单 `q` 退出
- **ACR 镜像推送（`bctrainctl acr push`）**：交互式 CLI 与 TUI 两种方式，移植自 `aliyun_acr_tools/acr_shell/acr_push.sh`；凭证默认取自 config，可命令行覆盖
- **额外文件/文件夹上传（`spec.extra_uploads`）**：把项目目录外的小数据集/测试数据随任务打包上传并在容器内解压到指定目录
- **配置项扩展**：新增 `wandb_api_key`、`swanlab_api_key`、`acr_registry`、`acr_namespace`、`acr_username`、`acr_password`（均脱敏存储）
- **配置命令增强**：`bctrainctl config` 直接打印当前配置并给出 Usage 提示；新增 `config set <key> <value>` 与 `config edit`
- 全局 `-h` 作为 `--help` 别名；新增 `tests/` 与打包过滤的单元测试

### 变更

- **打包过滤引擎重写**：改用 `pathspec`（与 Git 同款 gitwildmatch），完整支持 `**/x/*`、目录规则、`!` 取反；`.gitignore` 内的取反规则稳定生效，结果不再依赖目录遍历顺序
- **`init` 流程**：config 已存在时改为交互式询问是否覆盖（不再硬报错）；默认询问是否用 TUI 配置；连通性检查通过后才写入 config，避免失败留下半成品；可选填写 wandb / swanlab / ACR 凭证
- `bctrainctl --help` 中 `tui` 置于命令列表最上方
- `pathspec`、`textual` 由可选依赖提升为核心依赖

### 修复

- 遇到无效 AccessKey 等错误时只显示错误面板并以非零码退出，不再打印整段 Python 堆栈（`--debug` 仍可看堆栈）
- `bctrainctl acr` 及 `acr -h` 直接显示 Usage 而非报错
- TUI 提交界面的多行输入框（Exclude 等）此前高度为 0 导致无法输入，现已修复
- 主菜单 `q` 退出此前不生效，现已修复

## v0.1.x

早期版本：DLC 任务的提交、列表、查看、日志、停止、删除、同步，以及基础配置与代码打包上传。
