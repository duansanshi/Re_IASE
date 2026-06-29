#!/bin/bash
# 自动调度消融实验：等待GPU空闲后启动
# 使用 CUDA_VISIBLE_DEVICES 隔离GPU

PYTHON="/home/duanlei/anaconda3/envs/mambaimputer/bin/python"
WORKDIR="/home/duanlei/PriSTI"
cd "$WORKDIR"

# 定义实验：name | script | args | min_gpu_mem_gb
declare -a EXPERIMENTS=(
    # ===== 表6.13: 对完整模型的消融 =====
    # AQI v5: 线性插值预输入 (Linear + nofreq + curriculum)
    "aqi_v5_linear|NewImputer/UNET/aqi36_lightning_trainer_new_v5.py|--curriculum_temperature 0.5 --preimpute Linear --no_freq --exp_name aqi_v5_linear|12"
    # AQI v5: 移除频率截断 (Forward + no_freq + curriculum)
    "aqi_v5_nofreq|NewImputer/UNET/aqi36_lightning_trainer_new_v5.py|--curriculum_temperature 0.5 --no_freq --exp_name aqi_v5_nofreq|12"
    # AQI v5: 改用L1损失 (Forward + freq + no curriculum/smooth)
    "aqi_v5_l1|NewImputer/UNET/aqi36_lightning_trainer_new_v5.py|--curriculum_temperature 0.5 --curriculum_weight 0 --smooth_weight 0 --exp_name aqi_v5_l1|12"
    # PEMS08 v5: 线性插值预输入 (Linear + nofreq)
    "p08_v5_linear|NewImputer/UNET/pems08_lightning_trainer_v5.py|--curriculum_temperature 0.5 --smooth_weight 0.01 --preimpute Linear --no_freq --exp_name p08_v5_linear|20"
    # PEMS08 v5: 改用L1损失
    "p08_v5_l1|NewImputer/UNET/pems08_lightning_trainer_v5.py|--curriculum_temperature 0.5 --curriculum_weight 0 --smooth_weight 0 --exp_name p08_v5_l1|20"
    # PEMS04 v4: 线性插值预输入 (Linear + nofreq)
    "p04_v4_linear|NewImputer/UNET/pems04_lightning_trainer_v4.py|--preimpute Linear --no_freq --exp_name p04_v4_linear|24"
    # PEMS04 v4: 改用L1损失
    "p04_v4_l1|NewImputer/UNET/pems04_lightning_trainer_v4.py|--curriculum_weight 0 --smooth_weight 0 --exp_name p04_v4_l1|24"
)

# 获取完全空闲的GPU（无任何计算进程）
get_free_gpu() {
    local min_mem=$1
    # 获取所有有进程的GPU索引
    busy_gpus=$(nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader 2>/dev/null | sort -u)
    nvidia-smi --query-gpu=index,memory.total,gpu_bus_id --format=csv,noheader,nounits 2>/dev/null | while IFS=', ' read -r idx total bus_id; do
        total_gb=$((total / 1024))
        if [ "$total_gb" -ge "$min_mem" ]; then
            # 检查该GPU上是否有进程
            has_proc=$(nvidia-smi -i "$idx" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | head -1)
            if [ -z "$has_proc" ]; then
                echo "$idx"
                return
            fi
        fi
    done
}

echo "=== 消融实验自动调度器 ==="
echo "开始时间: $(date)"
echo "待运行实验: ${#EXPERIMENTS[@]} 个"
echo ""

exp_idx=0
while [ $exp_idx -lt ${#EXPERIMENTS[@]} ]; do
    IFS='|' read -r name script args min_mem <<< "${EXPERIMENTS[$exp_idx]}"
    
    # 检查是否已完成
    if [ -f "/tmp/${name}.log" ] && grep -q "Test Set Average MAE" "/tmp/${name}.log" 2>/dev/null; then
        echo "[$(date +%H:%M)] $name 已完成，跳过"
        exp_idx=$((exp_idx + 1))
        continue
    fi
    
    # 检查是否正在运行
    if screen -ls 2>/dev/null | grep -q "\.$name"; then
        echo "[$(date +%H:%M)] $name 正在运行中，跳过"
        exp_idx=$((exp_idx + 1))
        continue
    fi
    
    # 寻找空闲GPU
    free_gpu=$(get_free_gpu "$min_mem")
    
    if [ -n "$free_gpu" ]; then
        echo "[$(date +%H:%M)] 启动 $name 在 GPU${free_gpu} (需 ${min_mem}GB)"
        # 使用 CUDA_VISIBLE_DEVICES 隔离GPU，device参数固定为0
        screen -dmS "$name" bash -c "CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$free_gpu $PYTHON $script --device 0 $args 2>&1 | tee /tmp/${name}.log"
        sleep 120  # 等2分钟让GPU内存被占用
        exp_idx=$((exp_idx + 1))
    else
        sleep 120  # 没有空闲GPU，等2分钟后重试
    fi
done

echo ""
echo "=== 所有实验已启动 ==="
echo "结束时间: $(date)"
