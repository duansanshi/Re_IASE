"""Generate comparison table: Best Checkpoint vs Last Epoch"""
import csv

# Best checkpoint results (from experiment_results.csv)
best_ckpt = {
    # AQI36
    ("AQI36", "baseline"):    (8.8045, 296.82),
    ("AQI36", "v4_curr0.3"):  (8.7243, 284.80),
    ("AQI36", "v4_s03"):      (8.7132, 279.99),
    ("AQI36", "v4_s01"):      (8.6522, 277.61),
    ("AQI36", "v5_t1"):       (8.6571, 277.93),
    ("AQI36", "v5_t05"):      (8.6398, 277.26),
    ("AQI36", "v5_t05_s01"):  (8.7343, 295.10),
    ("AQI36", "v5_t03_s03"):  (8.8948, 307.42),
    ("AQI36", "v5_cw05"):     (8.7197, 289.86),
    ("AQI36", "v5_cw02"):     (8.9999, 316.87),
    # PEMS08
    ("PEMS08", "baseline"):   (9.6421, 280.52),
    ("PEMS08", "v4"):         (9.6236, 280.10),
    ("PEMS08", "v4_s03"):     (9.6186, 278.64),
    ("PEMS08", "v4_s01"):     (9.6063, 278.35),
    ("PEMS08", "v4_s005"):    (9.6319, 278.75),
    ("PEMS08", "v5_t05"):     (9.6166, 278.99),
    ("PEMS08", "v5_t1"):      (9.6371, 277.05),
    ("PEMS08", "v5_t05_s01"): (9.5973, 277.65),
    ("PEMS08", "v5_t05_s005"):(9.6097, 277.56),
    ("PEMS08", "v5_t03_s01"): (9.6053, 277.32),
    # PEMS04
    ("PEMS04", "baseline"):   (14.2745, 590.81),
    ("PEMS04", "v4"):         (14.1832, 578.46),
    ("PEMS04", "v4_s03"):     (14.2796, 587.93),
    ("PEMS04", "v4_s01"):     (14.2505, 584.74),
    ("PEMS04", "v4_cw05"):    (14.3160, 586.23),
    ("PEMS04", "v5_t05"):     (14.2678, 586.55),
    ("PEMS04", "v5_t05_s0"):  (14.1895, 579.72),
}

# Last epoch results (from batch retest)
last_epoch = {
    # AQI36 - baseline from retest_baseline.py
    ("AQI36", "baseline"):    (8.8045, 296.82),  # original was last epoch already
    ("AQI36", "v4_curr0.3"):  (8.8102, 309.60),
    ("AQI36", "v4_s03"):      (8.7442, 294.58),
    ("AQI36", "v4_s01"):      (8.6976, 292.51),
    ("AQI36", "v5_t1"):       (8.7244, 290.68),
    ("AQI36", "v5_t05"):      (8.6951, 286.25),
    ("AQI36", "v5_t05_s01"):  (8.8265, 301.55),
    ("AQI36", "v5_t03_s03"):  (8.8041, 299.64),
    ("AQI36", "v5_cw05"):     (8.7533, 291.25),
    ("AQI36", "v5_cw02"):     (8.8894, 311.56),
    # PEMS08
    ("PEMS08", "baseline"):   (9.6421, 280.52),  # original was last epoch already
    ("PEMS08", "v4"):         (9.5782, 277.85),
    ("PEMS08", "v4_s03"):     (9.5659, 276.61),
    ("PEMS08", "v4_s01"):     (9.5521, 275.94),
    ("PEMS08", "v4_s005"):    (9.5797, 276.20),
    ("PEMS08", "v5_t05"):     (9.5809, 279.42),
    ("PEMS08", "v5_t1"):      (9.5867, 277.78),
    ("PEMS08", "v5_t05_s01"): (9.5881, 278.94),
    ("PEMS08", "v5_t05_s005"):(9.5407, 276.07),
    ("PEMS08", "v5_t03_s01"): (9.5593, 276.84),
    # PEMS04
    ("PEMS04", "baseline"):   (14.2745, 590.81),  # original was last epoch already
    ("PEMS04", "v4"):         (14.1898, 580.58),
    ("PEMS04", "v4_s03"):     (14.2796, 587.93),
    ("PEMS04", "v4_s01"):     (14.2616, 589.17),
    ("PEMS04", "v4_cw05"):    (14.3160, 586.23),
    ("PEMS04", "v5_t05"):     (14.2678, 586.55),
    ("PEMS04", "v5_t05_s0"):  (14.1895, 579.72),
}

# Also the baseline best checkpoint retest results
baseline_best_ckpt = {
    ("AQI36", "baseline"):  (8.7425, 290.56),
    ("PEMS08", "baseline"): (9.8595, 284.36),
    ("PEMS04", "baseline"): (14.2745, 590.81),
}

# Generate comparison
print("=" * 120)
print(f"{'Dataset':<8} {'Experiment':<15} {'Best MAE':>10} {'Last MAE':>10} {'Diff':>8} {'Better':>8} | {'Best MSE':>10} {'Last MSE':>10} {'Diff':>8} {'Better':>8}")
print("=" * 120)

for dataset in ["AQI36", "PEMS08", "PEMS04"]:
    for key in sorted(best_ckpt.keys()):
        if key[0] != dataset:
            continue
        exp = key[1]
        b_mae, b_mse = best_ckpt[key]
        l_mae, l_mse = last_epoch[key]
        mae_diff = l_mae - b_mae
        mse_diff = l_mse - b_mse
        mae_better = "Last" if l_mae < b_mae else "Best"
        mse_better = "Last" if l_mse < b_mse else "Best"
        print(f"{dataset:<8} {exp:<15} {b_mae:>10.4f} {l_mae:>10.4f} {mae_diff:>+8.4f} {mae_better:>8} | {b_mse:>10.2f} {l_mse:>10.2f} {mse_diff:>+8.2f} {mse_better:>8}")
    print("-" * 120)

# Summary stats
print("\n=== SUMMARY: How often is Last Epoch better? ===")
for dataset in ["AQI36", "PEMS08", "PEMS04"]:
    mae_last_wins = 0
    mse_last_wins = 0
    total = 0
    for key in best_ckpt:
        if key[0] != dataset or key[1] == "baseline":
            continue
        total += 1
        b_mae, b_mse = best_ckpt[key]
        l_mae, l_mse = last_epoch[key]
        if l_mae < b_mae: mae_last_wins += 1
        if l_mse < b_mse: mse_last_wins += 1
    print(f"{dataset}: MAE Last wins {mae_last_wins}/{total}, MSE Last wins {mse_last_wins}/{total}")

# Find overall best per dataset (considering both strategies)
print("\n=== OVERALL BEST (considering both Best Ckpt and Last Epoch) ===")
for dataset in ["AQI36", "PEMS08", "PEMS04"]:
    all_results = []
    for key in best_ckpt:
        if key[0] != dataset:
            continue
        exp = key[1]
        b_mae, b_mse = best_ckpt[key]
        l_mae, l_mse = last_epoch[key]
        all_results.append((b_mae, f"{exp} (best_ckpt)", b_mse))
        all_results.append((l_mae, f"{exp} (last_epoch)", l_mse))
    if dataset in ["AQI36"]:
        # Also add baseline best ckpt
        bk = baseline_best_ckpt[(dataset, "baseline")]
        all_results.append((bk[0], "baseline (best_ckpt)", bk[1]))
    
    all_results.sort(key=lambda x: x[0])
    print(f"\n{dataset} Top 5 by MAE:")
    for mae, name, mse in all_results[:5]:
        print(f"  {name:<30} MAE={mae:.4f}  MSE={mse:.2f}")
