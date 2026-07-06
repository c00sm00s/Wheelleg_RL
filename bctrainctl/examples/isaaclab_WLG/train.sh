#!/usr/bin/env bash
set -euo pipefail

# IsaacLab 镜像没有系统 python，必须使用镜像内置的 python.sh
# 注意：alias 在非交互式脚本中不生效，所以这里用环境变量定义路径
ISAACLAB_PYTHON=/workspace/isaaclab/_isaac_sim/python.sh
ISAACLAB_PIP="$ISAACLAB_PYTHON -m pip"

export ISAACLAB_PATH=/workspace/isaaclab
export TZ=UTC
export PYTHONPATH="/workspace/code/source/Wheelleg:${PYTHONPATH:-}"

TASK_NAME="${TASK_NAME:-Wheelleg-Terrian-v0}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-2000}"
DEVICE="${DEVICE:-cuda:0}"

# 显示任务运行的基础环境信息
echo "[bctrainctl-wheelleg] start"
$ISAACLAB_PYTHON --version
$ISAACLAB_PIP --version
nvidia-smi || true
mkdir -p /workspace/output

# 设置网络代理
# echo "Network Proxy: ON"
# export http_proxy=http://10.0.0.152:7890
# export https_proxy=http://10.0.0.152:7890

# 取消DLC自动注入的Pytorch多机训练变量，避免isaaclab自动进多机训练模式
unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT NPROC_PER_NODE
$ISAACLAB_PYTHON -c "import os; print({k:os.environ.get(k) for k in ['RANK','WORLD_SIZE','LOCAL_RANK','MASTER_ADDR','MASTER_PORT','NPROC_PER_NODE','CUDA_VISIBLE_DEVICES']})"

# 任务运行脚本
cd /workspace/output

echo "[bctrainctl-wheelleg] code dir: /workspace/code"
echo "[bctrainctl-wheelleg] task: ${TASK_NAME}"
echo "[bctrainctl-wheelleg] num_envs: ${NUM_ENVS}"
echo "[bctrainctl-wheelleg] max_iterations: ${MAX_ITERATIONS}"
echo "[bctrainctl-wheelleg] device: ${DEVICE}"

$ISAACLAB_PYTHON /workspace/code/scripts/rsl_rl/train.py \
  --task="${TASK_NAME}" \
  --headless \
  --num_envs="${NUM_ENVS}" \
  --max_iterations="${MAX_ITERATIONS}" \
  --device="${DEVICE}"

# 预览任务输出的文件
echo "[bctrainctl-wheelleg] output files"
find /workspace/output -maxdepth 4 -type f | sort || true

echo "[bctrainctl-wheelleg] done"
