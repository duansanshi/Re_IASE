#!/bin/bash
# 监控所有消融实验进度
# 用法: watch -n 60 bash monitor_ablation.sh

echo "=== 消融实验监控 $(date +%H:%M:%S) ==="
echo ""

ALL_EXPS="p08_bl_freq p08_v5_freq p04_bl_freq p04_v4_freq aqi_linear aqi_linear_nofreq p08_backward_nofreq aqi_v5_linear aqi_v5_nofreq aqi_v5_l1 p08_v5_linear p08_v5_l1 p04_v4_linear p04_v4_l1 aqi_v5_l1_nofreq p08_v5_l1_nofreq p04_v4_l1_nofreq aqi_v5_backward p08_v5_forward p04_v4_forward aqi_v5_backward_nofreq p08_v5_forward_nofreq p04_v4_forward_nofreq"

echo "--- 表6.13 消融 (v5/v4) ---"

for f in $ALL_EXPS; do
    log="/tmp/${f}.log"
    if [ ! -f "$log" ]; then
        printf "%-25s 未启动\n" "$f:"
    elif grep -q "Test Set Average MAE" "$log" 2>/dev/null; then
        mae=$(grep "Test Set Average MAE" "$log" | awk '{print $NF}')
        mse=$(grep "Test Set Average MSE" "$log" | awk '{print $NF}')
        printf "%-25s ✅ MAE=%.4f MSE=%.2f\n" "$f:" "$mae" "$mse"
    elif grep -q "CUDA out of memory\|Error\|Traceback" "$log" 2>/dev/null; then
        printf "%-25s ❌ 失败\n" "$f:"
    else
        ep=$(tail -c 5000 "$log" 2>/dev/null | grep -oP 'Epoch \d+' | tail -1)
        printf "%-25s ⏳ %s\n" "$f:" "${ep:-初始化}"
    fi
done

echo ""
echo "调度器: $(screen -ls 2>/dev/null | grep ablation | awk '{print "运行中"}' || echo '未运行')"
