#!/bin/bash
# Batch retest all experiments with last.ckpt
# Runs sequentially per GPU, parallel across GPUs
cd /home/duanlei/PriSTI
PY=/home/duanlei/anaconda3/envs/mambaimputer/bin/python
SCRIPT=NewImputer/UNET/retest_last_epoch.py

# AQI36 experiments on GPU 0 (small, fast)
run_on_gpu() {
    local gpu=$1
    shift
    for exp in "$@"; do
        echo ">>> Running $exp on GPU $gpu"
        $PY $SCRIPT --exp $exp --device $gpu 2>&1 | grep -E "Retesting|MAE:|MSE:"
        echo "---"
    done
}

# GPU 0: AQI36 v4 experiments (9 experiments, ~30s each)
run_on_gpu 0 aqi_v4_curr03 aqi_v4_s03 aqi_v4_s01 aqi_v5_t1 aqi_v5_t05 aqi_v5_t05s01 aqi_v5_t03s03 aqi_v5_cw05 aqi_v5_cw02 &

# GPU 1: PEMS08 v4 experiments (4 experiments)
run_on_gpu 1 p08_v4 p08_v4_s03 p08_v4_s01 p08_v4_s005 &

# GPU 3: PEMS08 v5 experiments (5 experiments)
run_on_gpu 3 p08_v5_t05 p08_v5_t1 p08_v5_t05s01 p08_v5_t05s005 p08_v5_t03s01 &

# GPU 5: PEMS04 v4 experiments (4 experiments)
run_on_gpu 5 p04_v4 p04_v4_s03 p04_v4_s01 p04_v4_cw05 &

# GPU 4: PEMS04 v5 experiments (2 experiments)
run_on_gpu 4 p04_v5_t05 p04_v5_t05s0 &

wait
echo "=== ALL DONE ==="
