# 阿里云 API 与 RAM 权限

## API 权限点

下面这张表是当前代码实际会用到的云端能力。

| 云服务 | 权限点 / API | 被哪些命令使用 | 用途 |
| --- | --- | --- | --- |
| PAI DLC | `paidlc:ListJobs` / `ListJobs` | `init`、`list`、`list --remote`、`sync` | 联通性检查、列出任务、同步状态 |
| PAI DLC | `paidlc:CreateJob` / `CreateJob` | `submit` | 创建训练任务 |
| PAI DLC | `paidlc:GetJob` / `GetJob` | `logs` | 获取 Pod 信息 |
| PAI DLC | `paidlc:GetPodLogs` / `GetPodLogs` | `logs` | 拉取 Pod 日志 |
| PAI DLC | `paidlc:StopJob` / `StopJob` | `stop` | 停止任务 |
| PAI DLC | `paidlc:DeleteJob` / `DeleteJob` | `delete` | 删除任务 |
| PAI AIWorkspace | `paiworkspace:ListResources` / `ListResources` | `submit` | 把 quota 名称解析成工作空间绑定 `QuotaId` |
| OSS | `oss:GetBucketInfo` | `init` | 检查 bucket 可访问 |
| OSS | `oss:PutObject` | `submit` | 上传代码包，写挂载前缀占位对象 |
| OSS | `oss:GetObject` | `submit` 之后的容器运行阶段 | 容器通过签名 URL 下载代码包 |
| OSS | `oss:DeleteObject` | `delete` | 删除该任务对应的 OSS 代码包对象 |

说明：

- `show` 和 `config show` 当前不调用云端 API
- `submit --dry-run` 当前不会真正上传 OSS，也不会创建 DLC 任务
- `logs` 当前不拉事件，只拉第一个 Pod 的日志

## 推荐授权范围

调试阶段至少要保证：

- 当前 RAM 身份可以访问目标 DLC workspace
- 当前 RAM 身份可以访问目标 OSS bucket
- 当前 workspace 已绑定你要提交的 quota

如果你要先快速打通链路，RAM 权限点至少应覆盖：

```json
{
  "Version": "1",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "paidlc:ListJobs",
        "paidlc:CreateJob",
        "paidlc:GetJob",
        "paidlc:GetPodLogs",
        "paidlc:StopJob",
        "paidlc:DeleteJob",
        "paiworkspace:ListResources"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "oss:GetBucketInfo",
        "oss:PutObject",
        "oss:GetObject",
        "oss:DeleteObject"
      ],
      "Resource": "acs:oss:*:*:*"
    }
  ]
}
```

这是一份便于调试的宽松策略，不是最小资源范围策略。稳定后建议把 OSS Resource 收窄到实际 bucket 和 prefix。

## 官方文档参考

- PAI DLC RAM 权限总表：<https://help.aliyun.com/zh/pai/developer-reference/api-pai-dlc-2020-12-03-ram>
- DLC `CreateJob`：<https://help.aliyun.com/zh/pai/developer-reference/api-pai-dlc-2020-12-03-createjob>
- AIWorkspace `ListResources`：<https://help.aliyun.com/zh/pai/developer-reference/api-aiworkspace-2021-02-04-listresources>
- OSS 相关系统策略示例：<https://help.aliyun.com/zh/ram/developer-reference/aliyunpaidlcaccessingossrolepolicy>
