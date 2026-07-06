#!/usr/bin/env bash
set -euo pipefail

# IsaacLab 镜像没有系统 python，必须使用镜像内置的 python.sh
# 注意：alias 在非交互式脚本中不生效，所以这里用环境变量定义路径
ISAACLAB_PYTHON=/workspace/isaaclab/_isaac_sim/python.sh
ISAACLAB_PIP="$ISAACLAB_PYTHON -m pip"

export ISAACLAB_PATH=/workspace/isaaclab
export TZ=UTC

# 显示任务运行的基础环境信息
echo "[bctrainctl-isaaclab] start"
$ISAACLAB_PYTHON --version
$ISAACLAB_PIP --version
nvidia-smi || true
mkdir -p /workspace/output

# 设置网络代理
echo "Network Proxy: ON"
export http_proxy=http://10.0.0.152:7890
export https_proxy=http://10.0.0.152:7890

# 取消DLC自动注入的Pytorch多机训练变量，避免isaaclab自动进多机训练模式
unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT NPROC_PER_NODE
$ISAACLAB_PYTHON -c "import os; print({k:os.environ.get(k) for k in ['RANK','WORLD_SIZE','LOCAL_RANK','MASTER_ADDR','MASTER_PORT','NPROC_PER_NODE','CUDA_VISIBLE_DEVICES']})"

# 任务运行脚本
cd /workspace/output
$ISAACLAB_PYTHON /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Cartpole-v0 --headless

# 预览任务输出的文件
echo "[bctrainctl-isaaclab] output files"
find /workspace/output -maxdepth 4 -type f | sort || true

echo "[bctrainctl-isaaclab] done"

