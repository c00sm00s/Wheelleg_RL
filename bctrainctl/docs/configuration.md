# 配置文档

## 本地目录

默认本地目录是 `~/.bctrainctl/`，也可以显式设置 `BCTRAINCTL_HOME`。

目录结构：

```text
~/.bctrainctl/
  config.yaml
  jobs.db
  cache/
  packages/
  logs/
  templates/
```

各目录含义：

- `config.yaml`：本地配置，权限会尽量设为 `600`
- `jobs.db`：本地任务索引，记录 `job_id`、状态、代码包 URI、原始 spec 等
- `packages/`：本地打包出的 `tar.gz`
- `cache/`：预留缓存目录
- `logs/`：预留日志目录
- `templates/`：TUI 里保存的任务模板（YAML）

## 配置项

`config.yaml` 除阿里云凭证外，还支持以下可选字段（均可在 `bctrainctl init` 或 `bctrainctl config` 里设置）：

| 字段 | 说明 |
| --- | --- |
| `wandb_api_key` | Weights & Biases API Key，训练脚本可读取使用（脱敏存储） |
| `swanlab_api_key` | SwanLab API Key（脱敏存储） |
| `acr_registry` / `acr_namespace` | ACR 镜像仓库地址与命名空间，`bctrainctl acr push` 默认值 |
| `acr_username` / `acr_password` | ACR 登录凭证（密码脱敏存储） |

### 查看与修改配置

```bash
bctrainctl config                         # 直接打印当前配置（脱敏）并给出 Usage 提示
bctrainctl config show                    # 同上，仅打印配置
bctrainctl config set wandb_api_key xxxx  # 修改单个字段
bctrainctl config edit                    # 打开 TUI 配置界面交互编辑
```

`config set` 可修改的字段：`access_key_id`、`access_key_secret`、`region`、`workspace_id`、
`oss_bucket`、`oss_prefix`、`default_resource_id`、`default_image`、`wandb_api_key`、
`swanlab_api_key`、`acr_registry`、`acr_namespace`、`acr_username`、`acr_password`。

## 凭证来源

`bctrainctl` 支持两种凭证来源：

- 直接把 `access_key_id` / `access_key_secret` 保存到 `config.yaml`
- 复用 `~/.aliyun/config.json` 的当前 AK profile

当前解析顺序：

1. `config.yaml` 中显式指定的凭证
2. `config.yaml` 中指定的 `aliyun_profile`
3. 当前 `~/.aliyun/config.json` 的 `current` profile

## 初始化详情

```bash
bctrainctl init
```

初始化会引导填写：

- 阿里云凭证，或直接导入 `~/.aliyun/config.json` 当前 profile
- `region`
- `workspace_id`
- 默认 OSS bucket 和 prefix
- 默认 quota、镜像、资源规格

当前乌兰察布调试默认值：

- `region`: `cn-wulanchabu`
- `workspace_id`: `241942`
- `oss_bucket`: `oss-pai-wulanchabu-stark`
- `oss_prefix`: `bctrainctl`
- 默认 quota: `quota-4090`
- 默认镜像: `pytorch:2.7.0-gpu-py311-cu128-ubuntu24.04-accl-b244fc94-1764399419`
- 默认资源规格: `gpu=1 cpu=16 memory_gb=60`

任务优先级通过 `job.yaml` 里的 `spec.priority` 设置：

- 取值范围是 `1~9`
- `1` 最低
- `9` 最高
- 如果不写，DLC 默认优先级是 `1`

`init` 实际执行流程：

1. 检查 `~/.bctrainctl/config.yaml` 是否已存在；若已存在会**交互式询问是否覆盖**（而不是直接报错），也可用 `--force` 跳过询问
2. **默认询问「是否用 TUI 配置」**（回车即默认 Yes）：
   - 选 Yes → 打开 TUI 配置界面（首次配置会预填 init 默认参数），保存后同样做云端探测并打印结果
   - 选 No → 走下面的命令行逐项填写流程
3. 询问是否导入 `~/.aliyun/config.json` 当前 profile
4. 读取并校验输入项（包含可选的 wandb / swanlab / ACR 凭证）
5. 如果没有 `--skip-checks`，先做两次云端探测：调用 OSS `GetBucketInfo` 验证 bucket 可访问、调用 DLC `ListJobs` 验证 DLC OpenAPI 可访问
6. **探测通过后**才写入本地 `config.yaml`，因此连通性失败时不会留下半成品配置阻塞下一次重试

## 代码打包配置

`spec.code` 支持以下字段：

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `project_path` | string | 必填 | 代码相对路径，该路径下文件自动打包上传 |
| `exclude` | list[string] | `[]` | 代码打包排除的路径/模式 |
| `use_gitignore` | bool | `true` | 是否启用项目根目录 `.gitignore` 规则过滤打包文件 |
| `gitignore_include` | list[string] | `[]` | `.gitignore` 中匹配但仍需打包的路径（排除规则的白名单） |

无论 `use_gitignore` 是否开启，以下路径始终被排除：`.git`、`__pycache__`、`.venv`、`node_modules`、`outputs`、`wandb`。

示例：

```yaml
code:
  project_path: .
  exclude:
    - outputs
    - "*.log"
  use_gitignore: true          # 开启后自动读取 .gitignore 规则
  gitignore_include:           # .gitignore 中匹配但仍需打包的路径
    - build/weights
    - "*.so"
```

上面的配置表示：在 `exclude` 和 `.gitignore` 的基础上排除文件，但 `build/weights` 目录和所有 `.so` 文件即使被 `.gitignore` 忽略也会保留在代码包中。

打包过滤使用与 Git 相同的匹配引擎（基于 `pathspec` 的 gitwildmatch），因此支持完整的 gitignore 语法：

- `**/x/*`：任意层级下 `x/` 目录的直接子项
- `dir/`：仅匹配名为 `dir` 的目录及其内容
- `!keep.log`：except（取反）规则，把已被排除的路径重新保留
- `.gitignore` 内部的 `!` 取反规则会被正确解析（旧实现会忽略 `!` 行）

`gitignore_include` 以及 `.gitignore` 内的 `!` 取反规则会**稳定生效**：即使某条 except 规则位于被整体排除的目录下，也会按规则保留对应文件，结果不依赖打包时的目录遍历顺序。

## 额外文件/文件夹上传（extra_uploads）

`spec.extra_uploads` 用于把项目目录之外的小数据集或测试数据随任务一起上传，适合小体量数据或调试样本：

```yaml
spec:
  extra_uploads:
    - local: ./test_data       # 本地文件或目录（相对 job.yaml 所在目录解析）
      target: /workspace/data  # 容器内解压目标目录
    - local: ../fixtures/a.csv
      target: /workspace/fixtures
```

行为说明：

- 每个 `local` 会被单独打成 `tar.gz`，上传到 `oss://<bucket>/<oss_prefix>/extra_uploads/...`
- 容器启动时，代码包解压后会依次把每个额外包解压到对应的 `target` 目录
- `local` 为目录时，目录内容直接落到 `target` 下；为单个文件时，保留文件名放到 `target` 下
- 大数据集仍建议用 `spec.storage.mounts` 走 OSS/NAS 挂载，`extra_uploads` 只适合小数据

## 代码包与输出物

当前要区分三类文件：

- 本地代码包：保存在 `~/.bctrainctl/packages/*.tar.gz`
- OSS 代码包：保存在 `oss://<bucket>/<oss_prefix>/code_packages/...`
- 训练输出：保存在你显式挂载的 OSS/NAS 目标路径

重要行为：

- 任务结束后，容器内 `/tmp/<package>.tar.gz` 和 `workdir` 会随 Pod 一起消失
- OSS 里的代码包不会因任务结束自动删除
- 只有执行 `bctrainctl delete <job_id>` 时，才会顺手删除该任务对应的 OSS 代码包对象
- 训练输出不会被 `delete` 删除
