#!/bin/bash
# v6 早退出阈值扫描: 用同一个训练好的 checkpoint 测试不同阈值
# 用法: bash test_v6_exit_thresholds.sh <checkpoint_path> <device>

CKPT=${1:-"checkpoints/myimputer_aqi36_v6_baseline/best_model.ckpt"}
DEVICE=${2:-0}
PYTHON=/home/duanlei/anaconda3/envs/mambaimputer/bin/python

echo "Checkpoint: $CKPT"
echo "Device: $DEVICE"
echo "=============================="

for THR in 0.0 0.005 0.01 0.02 0.05 0.1 0.2 0.5; do
    echo ""
    echo ">>> Testing exit_threshold=${THR}"
    $PYTHON aqi36_lightning_trainer_v6.py \
        --device $DEVICE --test_only --no_freq \
        --checkpoint_path "$CKPT" \
        --exit_threshold $THR \
        --exp_name "v6_exit_${THR}" \
        2>&1 | grep -E "Test Set Average|Exit Statistics|Layer|Avg layers|Compute savings"
    echo "---"
done
