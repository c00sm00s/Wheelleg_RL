#!/usr/bin/env bash
set -euo pipefail

# 显示任务运行的基础环境信息
echo "[bctrainctl-yolo] start"
python --version
python -m pip --version
nvidia-smi || true
mkdir -p /workspace/output

# 安装所需要的额外库
python -m pip install --no-cache-dir ultralytics

# 设置网络代理
echo "Network Proxy: ON"
export http_proxy=http://10.0.0.152:7890
export https_proxy=http://10.0.0.152:7890

# 取消DLC自动注入的Pytorch多机训练变量，避免ultralytics自动进多机训练模式
unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT NPROC_PER_NODE
python -c "import os; print({k:os.environ.get(k) for k in ['RANK','WORLD_SIZE','LOCAL_RANK','MASTER_ADDR','MASTER_PORT','NPROC_PER_NODE','CUDA_VISIBLE_DEVICES']})"

# 任务运行脚本
yolo train \
  data=coco8.yaml \
  model=yolo26n.pt \
  epochs=10 \
  lr0=0.01 \
  project=/workspace/output \
  name=yolo-coco8-oss \
  exist_ok=True

# 预览任务输出的文件
echo "[bctrainctl-yolo] output files"
find /workspace/output -maxdepth 4 -type f | sort || true

echo "[bctrainctl-yolo] done"
