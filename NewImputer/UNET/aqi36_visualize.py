"""
AQI36 baseline 可视化脚本
加载 clean_baseline checkpoint，对 test 集前 N 个 batch 做 gt vs pred 可视化。
用法:
  python aqi36_visualize.py \
      --checkpoint_path /home/duanlei/PriSTI/checkpoints/myimputer_aqi36_clean_baseline/best_model.ckpt \
      --device 0 \
      --num_batches 5 \
      --save_dir ./pictures/aqi_baseline_vis
"""

import os
import sys
import argparse
import pickle

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from aqi36_lightning_datamodule import AQI36_DataModule
from imputer_lightning_module_new_v5 import aqi_lightning_module
from transformer_imputer_new import transformer_imputer


# ============================================================
# 可视化函数（原版，未修改）
# ============================================================
def visualize_pred_vs_gt(
    gt,
    pred,
    eval_mask,
    scaler,
    mean_scaler,
    batch_idx,
    save_dir="./pictures",
    sample_idx=0,
    prefix="ddpm",
    title_prefix="AQI DDPM",
    dpi=600,          # 新增：提高输出分辨率
    font_scale=3.0,   # 新增：整体放大字号
):
    """
    gt, pred, eval_mask:  shape = (B, L, K)
    gt / pred 是归一化后的值；scaler / mean_scaler 用于反归一化。
    eval_mask = 1 表示缺失且需要评估的位置。
    """
    os.makedirs(save_dir, exist_ok=True)

    # ===== 字号设置 =====
    title_fs = int(12 * font_scale)
    label_fs = int(10 * font_scale)
    tick_fs = int(9 * font_scale)
    cb_tick_fs = int(8 * font_scale)
    suptitle_fs = int(14 * font_scale)
    legend_fs = int(9 * font_scale)

    s = scaler.detach().cpu()
    m = mean_scaler.detach().cpu()

    gt_np   = (gt[sample_idx].detach().cpu() * s + m).numpy()
    pred_np = (pred[sample_idx].detach().cpu() * s + m).numpy()
    eval_np = eval_mask[sample_idx].detach().cpu().numpy()

    L, K = gt_np.shape

    gt_missing   = np.where(eval_np > 0, gt_np,   np.nan)
    pred_missing = np.where(eval_np > 0, pred_np, np.nan)
    err_missing  = np.where(eval_np > 0, np.abs(gt_np - pred_np), np.nan)

    vmin = np.nanmin(gt_missing) if np.any(eval_np > 0) else 0.0
    vmax = np.nanmax(gt_missing) if np.any(eval_np > 0) else 1.0

    # ===== 热力图：放大画布 =====
    fig, axes = plt.subplots(1, 3, figsize=(26, 7))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.82, bottom=0.14, wspace=0.35)

    def _imshow(ax, data, title, cmap, vmin=None, vmax=None):
        im = ax.imshow(
            data.T,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest"
        )
        ax.set_title(title, fontsize=title_fs, pad=10)
        ax.set_xlabel("Time step", fontsize=label_fs)
        ax.set_ylabel("Sensor", fontsize=label_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)

        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=cb_tick_fs)

    _imshow(axes[0], gt_missing,   "Ground Truth (missing only)", "viridis", vmin, vmax)
    _imshow(axes[1], pred_missing, "Prediction (missing only)",   "viridis", vmin, vmax)
    _imshow(axes[2], err_missing,  "|Error| (missing only)",      "Reds")

    fig.suptitle(
        f"{title_prefix} - Batch {batch_idx}",
        fontsize=suptitle_fs,
        fontweight="bold"
    )

    plt.savefig(
        os.path.join(save_dir, f"{prefix}_heatmap_batch_{batch_idx}.pdf"),
        dpi=dpi,
        bbox_inches="tight"
    )
    plt.close(fig)

    sensor_dir = os.path.join(save_dir, f"{prefix}_sensors_batch_{batch_idx}")
    os.makedirs(sensor_dir, exist_ok=True)

    time_axis = np.arange(L)

    for k in range(K):
        mask_k = eval_np[:, k].astype(bool)
        if mask_k.sum() == 0:
            continue

        # ===== 单传感器曲线图：放大画布 =====
        fig_s, ax = plt.subplots(figsize=(12, 5.5))
        fig_s.subplots_adjust(left=0.10, right=0.97, top=0.84, bottom=0.16)

        ax.plot(
            time_axis,
            gt_np[:, k],
            color="gray",
            linewidth=1.4,
            alpha=0.5,
            label="GT (all)"
        )

        gt_miss   = np.where(mask_k, gt_np[:, k],   np.nan)
        pred_miss = np.where(mask_k, pred_np[:, k], np.nan)

        ax.plot(
            time_axis,
            gt_miss,
            "o-",
            color="tab:blue",
            markersize=7,
            linewidth=2.0,
            label="GT (missing)"
        )
        ax.plot(
            time_axis,
            pred_miss,
            "s--",
            color="tab:red",
            markersize=7,
            linewidth=2.0,
            label="Pred (missing)"
        )

        for t in range(L):
            if mask_k[t]:
                ax.axvspan(t - 0.5, t + 0.5, color="lightyellow", alpha=0.4)

        mae_k = np.nanmean(np.abs(gt_np[:, k] - pred_np[:, k])[mask_k])

        ax.set_title(
            f"Sensor {k}  (MAE={mae_k:.2f}, #miss={int(mask_k.sum())})",
            fontsize=title_fs,
            pad=8
        )
        ax.set_xlabel("Time step", fontsize=label_fs)
        ax.set_ylabel("Value", fontsize=label_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)

        ax.legend(fontsize=legend_fs, loc="upper right", framealpha=0.8)
        ax.grid(True, alpha=0.3)
        ax.margins(x=0.01)

        plt.savefig(
            os.path.join(sensor_dir, f"sensor_{k}.pdf"),
            dpi=dpi,
            bbox_inches="tight"
        )
        plt.close(fig_s)

    print(f"[Vis] Saved batch {batch_idx} to {save_dir}")
# def visualize_pred_vs_gt(
#     gt,
#     pred,
#     eval_mask,
#     scaler,
#     mean_scaler,
#     batch_idx,
#     save_dir="./pictures",
#     sample_idx=0,
#     prefix="ddpm",
#     title_prefix="AQI DDPM",
# ):
#     """
#     gt, pred, eval_mask:  shape = (B, L, K)
#     gt / pred 是归一化后的值；scaler / mean_scaler 用于反归一化。
#     eval_mask = 1 表示缺失且需要评估的位置。
#     """
#     os.makedirs(save_dir, exist_ok=True)

#     s = scaler.detach().cpu()
#     m = mean_scaler.detach().cpu()

#     gt_np   = (gt[sample_idx].detach().cpu() * s + m).numpy()      # (L, K)
#     pred_np = (pred[sample_idx].detach().cpu() * s + m).numpy()    # (L, K)
#     eval_np = eval_mask[sample_idx].detach().cpu().numpy()          # (L, K)

#     L, K = gt_np.shape

#     gt_missing   = np.where(eval_np > 0, gt_np,   np.nan)
#     pred_missing = np.where(eval_np > 0, pred_np,  np.nan)
#     err_missing  = np.where(eval_np > 0, np.abs(gt_np - pred_np), np.nan)

#     vmin = np.nanmin(gt_missing) if np.any(eval_np > 0) else 0.0
#     vmax = np.nanmax(gt_missing) if np.any(eval_np > 0) else 1.0

#     fig, axes = plt.subplots(1, 3, figsize=(20, 5))
#     fig.subplots_adjust(left=0.06, right=0.97, top=0.82, bottom=0.14, wspace=0.35)

#     def _imshow(ax, data, title, cmap, vmin=None, vmax=None):
#         im = ax.imshow(data.T, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
#         ax.set_title(title, fontsize=12, pad=8)
#         ax.set_xlabel("Time step", fontsize=10)
#         ax.set_ylabel("Sensor", fontsize=10)
#         cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#         cb.ax.tick_params(labelsize=8)

#     _imshow(axes[0], gt_missing,   "Ground Truth (missing only)", "viridis", vmin, vmax)
#     _imshow(axes[1], pred_missing, "Prediction (missing only)",  "viridis", vmin, vmax)
#     _imshow(axes[2], err_missing,  "|Error| (missing only)",     "Reds")

#     fig.suptitle(f"{title_prefix} - Batch {batch_idx}", fontsize=14, fontweight="bold")
#     plt.savefig(os.path.join(save_dir, f"{prefix}_heatmap_batch_{batch_idx}.pdf"), dpi=200)
#     plt.close(fig)

#     sensor_dir = os.path.join(save_dir, f"{prefix}_sensors_batch_{batch_idx}")
#     os.makedirs(sensor_dir, exist_ok=True)

#     time_axis = np.arange(L)
#     for k in range(K):
#         mask_k = eval_np[:, k].astype(bool)
#         if mask_k.sum() == 0:
#             continue

#         fig_s, ax = plt.subplots(figsize=(9, 4))
#         fig_s.subplots_adjust(left=0.09, right=0.97, top=0.85, bottom=0.16)

#         ax.plot(time_axis, gt_np[:, k], color="gray", linewidth=1.0, alpha=0.5, label="GT (all)")

#         gt_miss   = np.where(mask_k, gt_np[:, k],   np.nan)
#         pred_miss = np.where(mask_k, pred_np[:, k],  np.nan)

#         ax.plot(time_axis, gt_miss,   "o-",  color="tab:blue", markersize=5, linewidth=1.5, label="GT (missing)")
#         ax.plot(time_axis, pred_miss, "s--", color="tab:red",  markersize=5, linewidth=1.5, label="Pred (missing)")

#         for t in range(L):
#             if mask_k[t]:
#                 ax.axvspan(t - 0.5, t + 0.5, color="lightyellow", alpha=0.4)

#         mae_k = np.nanmean(np.abs(gt_np[:, k] - pred_np[:, k])[mask_k])
#         ax.set_title(f"Sensor {k}  (MAE={mae_k:.2f}, #miss={int(mask_k.sum())})", fontsize=12, pad=6)
#         ax.set_xlabel("Time step", fontsize=10)
#         ax.set_ylabel("Value", fontsize=10)
#         ax.legend(fontsize=9, loc="upper right", framealpha=0.8)
#         ax.grid(True, alpha=0.3)
#         ax.margins(x=0.01)

#         plt.savefig(os.path.join(sensor_dir, f"sensor_{k}.pdf"), dpi=200)
#         plt.close(fig_s)

#     print(f"[Vis] Saved batch {batch_idx} to {save_dir}")


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str,
                        default="/home/duanlei/PriSTI/checkpoints/myimputer_aqi36_clean_baseline/best_model.ckpt")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num_batches", type=int, default=5,
                        help="可视化 test 集前 N 个 batch（-1 表示全部）")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="batch 内哪个样本画 heatmap")
    parser.add_argument("--save_dir", type=str, default="./pictures/aqi_baseline_vis")
    parser.add_argument("--prefix", type=str, default="aqi_baseline")
    parser.add_argument("--title_prefix", type=str, default="AQI36 Baseline")
    parser.add_argument("--preimpute", type=str, default="Forward")
    parser.add_argument("--no_freq", action="store_true")
    args = parser.parse_args()

    device_str = f"cuda:{args.device}"
    torch.set_grad_enabled(False)

    # ---------- 模型 ----------
    model = transformer_imputer(device=device_str)
    lit_model = aqi_lightning_module.load_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        model=model,
        lr=5e-4,
    )
    lit_model.eval()
    lit_model.to(device_str)

    # ---------- 数据 ----------
    dm = AQI36_DataModule(
        device=device_str,
        preimpute_flag=args.preimpute,
        freq_flag=not args.no_freq,
    )
    dm.setup()
    test_loader = dm.test_dataloader()

    # ---------- scaler ----------
    path = "/home/duanlei/PriSTI/data/pm25/pm25_meanstd.pk"
    with open(path, "rb") as f:
        train_mean, train_std = pickle.load(f)
    scaler      = torch.from_numpy(train_std).float().to(device_str)
    mean_scaler = torch.from_numpy(train_mean).float().to(device_str)

    # ---------- 推理 & 可视化 ----------
    for batch_idx, batch in enumerate(test_loader):
        if args.num_batches >= 0 and batch_idx >= args.num_batches:
            break

        (
            observed_data, observed_mask, observed_tp,
            gt_mask, for_pattern_mask, cut_length,
            coeffs, cond_mask,
        ) = lit_model.model.process_data(batch)

        imputed_results, _ = lit_model.model(coeffs, cond_mask)

        # (B, K, L) → (B, L, K)
        imputed_results = imputed_results.permute(0, 2, 1)
        eval_mask_t     = (observed_mask - cond_mask).permute(0, 2, 1)
        observed_data_t = observed_data.permute(0, 2, 1)

        visualize_pred_vs_gt(
            gt=observed_data_t,
            pred=imputed_results,
            eval_mask=eval_mask_t,
            scaler=scaler,
            mean_scaler=mean_scaler,
            batch_idx=batch_idx,
            save_dir=args.save_dir,
            sample_idx=args.sample_idx,
            prefix=args.prefix,
            title_prefix=args.title_prefix,
        )

    print(f"\n[Done] 可视化结果已保存至 {args.save_dir}")


if __name__ == "__main__":
    main()
