#!/usr/bin/env bash
set -euo pipefail

# Reproduce the best IASE-Net results recorded in experiment_results.csv.
#
# Usage:
#   bash reproduce_best_results.sh commands [all|aqi36|pems04|pems08] [gpu]
#   bash reproduce_best_results.sh train    [all|aqi36|pems04|pems08] [gpu]
#   bash reproduce_best_results.sh test-best [all|aqi36|pems04|pems08] [gpu]
#
# Default action is "commands" so this script does not accidentally launch
# long training jobs.

ACTION="${1:-commands}"
TARGET="${2:-all}"
GPU="${3:-0}"

ROOT="/home/duanlei/PriSTI"
UNET="${ROOT}/NewImputer/UNET"
CONDA_SH="/home/duanlei/anaconda3/etc/profile.d/conda.sh"
ENV_NAME="mambaimputer"

cd "${UNET}"
source "${CONDA_SH}"
conda activate "${ENV_NAME}"

run_or_print() {
    local cmd="$1"
    if [[ "${ACTION}" == "commands" ]]; then
        printf '%s\n\n' "${cmd}"
    else
        eval "${cmd}"
    fi
}

require_ckpt() {
    local ckpt="$1"
    if [[ ! -f "${ckpt}" ]]; then
        echo "Missing checkpoint: ${ckpt}" >&2
        exit 1
    fi
}

train_aqi36() {
    # Best in experiment_results.csv:
    # AQI36,v5_t05,MAE=8.6398,MSE=277.26,soft curriculum t=0.5 s=0.03
    run_or_print "python aqi36_lightning_trainer_new_v5.py \
--device ${GPU} \
--epochs 200 \
--lr 5e-4 \
--preimpute Forward \
--freq_ratio 0.1 \
--curriculum_weight 0.3 \
--curriculum_warmup_ratio 0.1 \
--curriculum_temperature 0.5 \
--smooth_weight 0.03 \
--exp_name v5_soft_t0.5_c0.3_s0.03"
}

train_pems08() {
    # Best in experiment_results.csv:
    # PEMS08,v5_t05_s01,MAE=9.5973,MSE=277.65,soft curriculum t=0.5 s=0.01
    run_or_print "python pems08_lightning_trainer_v5.py \
--device ${GPU} \
--epochs 40 \
--lr 5e-4 \
--preimpute Backward \
--curriculum_weight 0.3 \
--curriculum_warmup_ratio 0.1 \
--curriculum_temperature 0.5 \
--smooth_weight 0.01 \
--limit_train_batches 780 \
--exp_name p08_v5_t05_s01"
}

train_pems04() {
    # Best in experiment_results.csv:
    # PEMS04,v4,MAE=14.1832,MSE=578.46,hard curriculum cw=0.3 s=0
    run_or_print "python pems04_lightning_trainer_v4.py \
--device ${GPU} \
--epochs 40 \
--lr 5e-4 \
--preimpute Backward \
--curriculum_weight 0.3 \
--curriculum_warmup_ratio 0.1 \
--smooth_weight 0.0 \
--limit_train_batches 742 \
--exp_name pems04_v4_curr0.3"
}

test_aqi36_best() {
    local ckpt="${UNET}/checkpoints/myimputer_aqi36_v5_soft_t0.5_c0.3_s0.03/best_model.ckpt"
    require_ckpt "${ckpt}"
    run_or_print "python aqi36_lightning_trainer_new_v5.py \
--device ${GPU} \
--test_only \
--checkpoint_path ${ckpt} \
--preimpute Forward \
--freq_ratio 0.1 \
--curriculum_weight 0.3 \
--curriculum_warmup_ratio 0.1 \
--curriculum_temperature 0.5 \
--smooth_weight 0.03 \
--exp_name v5_soft_t0.5_c0.3_s0.03"
}

test_pems08_best() {
    local ckpt="${ROOT}/checkpoints/myimputer_p08_v5_t05_s01/best_model.ckpt"
    require_ckpt "${ckpt}"
    run_or_print "python pems08_lightning_trainer_v5.py \
--device ${GPU} \
--test_only \
--checkpoint_path ${ckpt} \
--preimpute Backward \
--curriculum_weight 0.3 \
--curriculum_warmup_ratio 0.1 \
--curriculum_temperature 0.5 \
--smooth_weight 0.01 \
--limit_train_batches 780 \
--exp_name p08_v5_t05_s01"
}

test_pems04_best() {
    local ckpt="${ROOT}/checkpoints/myimputer_pems04_v4_curr0.3/best_model.ckpt"
    require_ckpt "${ckpt}"
    run_or_print "python pems04_lightning_trainer_v4.py \
--device ${GPU} \
--test_only \
--checkpoint_path ${ckpt} \
--preimpute Backward \
--curriculum_weight 0.3 \
--curriculum_warmup_ratio 0.1 \
--smooth_weight 0.0 \
--limit_train_batches 742 \
--exp_name pems04_v4_curr0.3"
}

select_targets() {
    case "${TARGET}" in
        all)
            printf '%s\n' aqi36 pems04 pems08
            ;;
        aqi36|pems04|pems08)
            printf '%s\n' "${TARGET}"
            ;;
        *)
            echo "Unknown target: ${TARGET}" >&2
            echo "Expected one of: all, aqi36, pems04, pems08" >&2
            exit 1
            ;;
    esac
}

case "${ACTION}" in
    commands|train|test-best)
        ;;
    *)
        echo "Unknown action: ${ACTION}" >&2
        echo "Expected one of: commands, train, test-best" >&2
        exit 1
        ;;
esac

for dataset in $(select_targets); do
    case "${ACTION}:${dataset}" in
        commands:aqi36|train:aqi36) train_aqi36 ;;
        commands:pems04|train:pems04) train_pems04 ;;
        commands:pems08|train:pems08) train_pems08 ;;
        test-best:aqi36) test_aqi36_best ;;
        test-best:pems04) test_pems04_best ;;
        test-best:pems08) test_pems08_best ;;
    esac
done
