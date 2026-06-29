"""
Convert the user's AQI36 / PEMS04 / PEMS08 datasets to PyPOTS-compatible format.

The conversion follows the dataset protocols in the uploaded dataset_aqi.py,
dataset_pems04.py, and dataset_pems08.py:
  - PEMS04/PEMS08: eval_length=24, val_len=0.1, test_len=0.2,
    ob_mask=(df.values != 0), seed=9101112, block/point missing rules.
  - AQI36: eval_length=36, target_dim=36, month-based split,
    ground file pm25_ground.txt, missing-pattern file pm25_missing.txt.
  - Standardization follows the original code: (value - train_mean) / train_std.

Outputs for each dataset:
  <out_dir>/<dataset>/train_set.npz    # contains X only, for PyPOTS training
  <out_dir>/<dataset>/val_set.npz      # contains X, X_ori, eval_mask, cut_length
  <out_dir>/<dataset>/test_set.npz     # contains X, X_ori, eval_mask, cut_length
  <out_dir>/<dataset>/train.h5         # contains key X, can be passed to PyPOTS directly
  <out_dir>/<dataset>/val.h5           # contains key X, can be passed to PyPOTS directly
  <out_dir>/<dataset>/test.h5          # contains key X, can be passed to PyPOTS directly
  <out_dir>/<dataset>/meta.npz         # mean, std, config metadata

Notes:
  - PyPOTS expects X with shape [n_samples, n_steps, n_features].
  - Missing entries in X are represented by np.nan.
  - X_ori is the standardized reference series. It is used only for external MAE.
  - eval_mask marks the positions to evaluate MAE on. It follows the original
    protocol: eval_mask = observed_mask - cond_mask, clipped to {0,1}.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import h5py
except ImportError:  # h5 output becomes optional
    h5py = None


# ----------------------------- common utilities -----------------------------


def safe_std(std: np.ndarray) -> np.ndarray:
    std = np.asarray(std, dtype=np.float32).copy()
    std[std == 0] = 1.0
    return std


def load_pickle_mean_std(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        mean, std = pickle.load(f)
    return np.asarray(mean, dtype=np.float32), safe_std(np.asarray(std, dtype=np.float32))


def save_pickle_mean_std(path: Path, mean: np.ndarray, std: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump((mean, std), f)


def save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def save_h5_x(path: Path, X: np.ndarray) -> None:
    if h5py is None:
        print(f"[WARN] h5py is not installed; skip HDF5 output: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=X.astype(np.float32), compression="gzip", compression_opts=4)


def nanify_by_mask(data_std: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
    """Return PyPOTS input X: standardized values with np.nan at missing/hidden positions."""
    X = data_std.astype(np.float32).copy()
    X[keep_mask.astype(bool) == 0] = np.nan
    return X


def make_windows(
    data_std: np.ndarray,
    cond_mask: np.ndarray,
    observed_mask: np.ndarray,
    eval_length: int,
    mode: str,
) -> Dict[str, np.ndarray]:
    """
    Build fixed-length windows following the user's Dataset.__getitem__ protocol.

    For train/valid: sliding windows with stride 1.
    For test: non-overlapping windows with an additional tail window; the first
    cut_length positions in the tail window are excluded from MAE to avoid double counting.
    """
    T, F = data_std.shape
    current_length = T - eval_length + 1
    if current_length <= 0:
        raise ValueError(f"sequence length {T} is shorter than eval_length {eval_length}")

    if mode == "test":
        n_sample = T // eval_length
        starts = list(np.arange(0, eval_length * n_sample, eval_length, dtype=int))
        cut_lengths = [0] * len(starts)
        remainder = T % eval_length
        if remainder != 0:
            starts.append(current_length - 1)
            cut_lengths.append(eval_length - remainder)
    else:
        starts = list(np.arange(current_length, dtype=int))
        cut_lengths = [0] * len(starts)

    X_list, X_ori_list, eval_mask_list = [], [], []
    for start, cut_len in zip(starts, cut_lengths):
        end = start + eval_length
        w_data = data_std[start:end]
        w_cond = cond_mask[start:end]
        w_ob = observed_mask[start:end]

        # Original code uses eval_mask = ob_mask - cond_mask, then clips to [0, 1].
        w_eval = np.clip(w_ob.astype(np.float32) - w_cond.astype(np.float32), 0, 1)
        if cut_len > 0:
            w_eval[:cut_len] = 0.0

        X_list.append(nanify_by_mask(w_data, w_cond))
        X_ori_list.append(nanify_by_mask(w_data, w_ob))
        eval_mask_list.append(w_eval.astype(np.float32))

    return {
        "X": np.stack(X_list, axis=0).astype(np.float32),
        "X_ori": np.stack(X_ori_list, axis=0).astype(np.float32),
        "eval_mask": np.stack(eval_mask_list, axis=0).astype(np.float32),
        "cut_length": np.asarray(cut_lengths, dtype=np.int32),
        "starts": np.asarray(starts, dtype=np.int32),
    }


def summarize_split(name: str, split: Dict[str, np.ndarray]) -> Dict[str, float]:
    X = split["X"]
    eval_mask = split.get("eval_mask")
    total = float(np.prod(X.shape))
    missing = float(np.isnan(X).sum())
    summary = {
        "shape": list(X.shape),
        "input_missing_ratio": missing / total,
    }
    if eval_mask is not None:
        summary["eval_points"] = int(eval_mask.sum())
        summary["eval_ratio"] = float(eval_mask.sum() / total)
    print(f"[{name}] {json.dumps(summary, ensure_ascii=False)}")
    return summary


def write_dataset_output(out_base: Path, splits: Dict[str, Dict[str, np.ndarray]], meta: Dict) -> None:
    out_base.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for split_name, split in splits.items():
        summaries[split_name] = summarize_split(split_name, split)
        save_npz(out_base / f"{split_name}_set.npz", **split)
        save_h5_x(out_base / f"{split_name}.h5", split["X"])

    # Save metadata. Arrays are stored separately to avoid object dtype where possible.
    meta_to_json = dict(meta)
    meta_to_json["summaries"] = summaries
    with open(out_base / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_to_json, f, indent=2, ensure_ascii=False)

    arrays = {}
    if "mean" in meta:
        arrays["mean"] = np.asarray(meta["mean"], dtype=np.float32)
    if "std" in meta:
        arrays["std"] = np.asarray(meta["std"], dtype=np.float32)
    if arrays:
        save_npz(out_base / "meta_arrays.npz", **arrays)


# ----------------------------- PeMS conversion ------------------------------


def sample_mask(shape, p=0.0015, p_noise=0.05, max_seq=1, min_seq=1, rng=None):
    """Same mask generator as dataset_pems04.py / dataset_pems08.py."""
    if rng is None:
        rand = np.random.random
        randint = np.random.randint
    else:
        rand = rng.random
        randint = rng.integers

    mask = rand(shape) < p
    for col in range(mask.shape[1]):
        idxs = np.flatnonzero(mask[:, col])
        if not len(idxs):
            continue
        fault_len = min_seq
        if max_seq > min_seq:
            fault_len = fault_len + int(randint(max_seq - min_seq))
        idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
        idxs = np.unique(idxs_ext)
        idxs = np.clip(idxs, 0, shape[0] - 1)
        mask[idxs, col] = True
    mask = mask | (rand(mask.shape) < p_noise)
    return mask.astype("uint8")


def build_pems(
    dataset: str,
    data_root: Path,
    out_dir: Path,
    eval_length: int = 24,
    val_len: float = 0.1,
    test_len: float = 0.2,
    missing_pattern: str = "block",
    missing_ratio: float | None = None,
    seed: int = 9101112,
) -> None:
    dataset = dataset.lower()
    if dataset not in {"pems04", "pems08"}:
        raise ValueError(dataset)

    folder = "PEMS04" if dataset == "pems04" else "PEMS08"
    h5_path = data_root / folder / f"{dataset}.h5"
    meanstd_path = data_root / folder / f"{dataset}_meanstd.pk"

    if not h5_path.exists():
        raise FileNotFoundError(f"Cannot find {h5_path}")

    df = pd.read_hdf(h5_path)
    values = df.fillna(0).values.astype(np.float32)
    T, F = values.shape

    # Follow the user's get_mean_std(): first 70% values, including zeros.
    if meanstd_path.exists():
        mean, std = load_pickle_mean_std(meanstd_path)
    else:
        train_data = values[: int(T * 0.7)]
        mean = train_data.mean(axis=0).astype(np.float32)
        std = safe_std(train_data.std(axis=0).astype(np.float32))
        save_pickle_mean_std(meanstd_path, mean, std)

    ob_mask = (df.values != 0.0).astype("uint8")
    rng = np.random.default_rng(seed)

    if missing_pattern == "block":
        p = 0.0015 if missing_ratio is None else missing_ratio
        eval_mask_full = sample_mask((T, F), p=p, p_noise=0.05, min_seq=12, max_seq=12 * 4, rng=rng)
    elif missing_pattern == "point":
        # Same as user's code: point missing uses p_noise=0.25.
        p_noise = 0.25 if missing_ratio is None else missing_ratio
        eval_mask_full = sample_mask((T, F), p=0.0, p_noise=p_noise, min_seq=12, max_seq=12 * 4, rng=rng)
    else:
        raise ValueError("missing_pattern must be 'block' or 'point'")

    gt_mask_full = (1 - (eval_mask_full | (1 - ob_mask))).astype("uint8")
    data_std = ((values - mean) / std).astype(np.float32)

    val_start = int((1 - val_len - test_len) * T)
    test_start = int((1 - test_len) * T)

    ranges = {
        "train": slice(0, val_start),
        "val": slice(val_start, test_start),
        "test": slice(test_start, T),
    }

    splits: Dict[str, Dict[str, np.ndarray]] = {}
    for mode, sl in ranges.items():
        if mode == "train":
            # Static PyPOTS input: only real unavailable points are NaN.
            # Training-time artificial masking is handled by PyPOTS internally.
            cond = ob_mask[sl]
        else:
            # Validation/test follow the original code: cond_mask = gt_mask.
            cond = gt_mask_full[sl]
        splits[mode] = make_windows(data_std[sl], cond, ob_mask[sl], eval_length, mode)

    meta = {
        "dataset": dataset,
        "raw_h5": str(h5_path),
        "meanstd_path": str(meanstd_path),
        "mean": mean,
        "std": std,
        "eval_length": eval_length,
        "val_len": val_len,
        "test_len": test_len,
        "missing_pattern": missing_pattern,
        "missing_ratio": missing_ratio,
        "seed": seed,
        "raw_shape": [int(T), int(F)],
        "split_indices": {"val_start": int(val_start), "test_start": int(test_start)},
    }
    write_dataset_output(out_dir / dataset, splits, meta)


# ----------------------------- AQI36 conversion -----------------------------


def build_aqi36(
    aqi_root: Path,
    out_dir: Path,
    eval_length: int = 36,
    target_dim: int = 36,
    val_len: float = 0.1,
    mask_sensor: List[int] | None = None,
) -> None:
    """Build AQI36 exactly following the month split in dataset_aqi.py."""
    mask_sensor = [] if mask_sensor is None else list(mask_sensor)

    meanstd_path = aqi_root / "pm25_meanstd.pk"
    ground_path = aqi_root / "SampleData" / "pm25_ground.txt"
    missing_path = aqi_root / "SampleData" / "pm25_missing.txt"

    if not meanstd_path.exists():
        raise FileNotFoundError(
            f"Cannot find {meanstd_path}. To follow your code strictly, this file must be provided."
        )
    if not ground_path.exists():
        raise FileNotFoundError(f"Cannot find {ground_path}")
    if not missing_path.exists():
        raise FileNotFoundError(f"Cannot find {missing_path}")

    mean, std = load_pickle_mean_std(meanstd_path)
    df = pd.read_csv(ground_path, index_col="datetime", parse_dates=True)
    df_gt = pd.read_csv(missing_path, index_col="datetime", parse_dates=True)

    if df.shape[1] != target_dim:
        print(f"[WARN] AQI36 target_dim={target_dim}, but loaded {df.shape[1]} columns.")

    def build_month_split(mode: str) -> Dict[str, np.ndarray]:
        if mode == "train":
            month_list = [1, 2, 4, 5, 7, 8, 10, 11]
        elif mode == "val":
            month_list = [2, 5, 8, 11]
        elif mode == "test":
            month_list = [3, 6, 9, 12]
        else:
            raise ValueError(mode)

        X_all, X_ori_all, eval_all, cut_all, start_all, month_all = [], [], [], [], [], []

        for month_idx, month in enumerate(month_list):
            current_df = df[df.index.month == month]
            current_df_gt = df_gt[df_gt.index.month == month]

            if mode == "train" and month in [2, 5, 8, 11]:
                cut_len = int(val_len * len(current_df))
                current_df = current_df[:-cut_len]
                current_df_gt = current_df_gt[:-cut_len]
            if mode == "val":
                cut_len = int(val_len * len(current_df))
                current_df = current_df[-cut_len:]
                current_df_gt = current_df_gt[-cut_len:]

            c_mask = (1 - current_df.isnull().values).astype("uint8")
            c_gt_mask = (1 - current_df_gt.isnull().values).astype("uint8")

            if len(mask_sensor) > 0:
                for sensor in mask_sensor:
                    c_gt_mask[:, sensor] = 0
                    if mode == "train":
                        c_mask[:, sensor] = 0

            values = current_df.fillna(0).values.astype(np.float32)
            data_std = ((values - mean) / std).astype(np.float32)

            # Static PyPOTS input: train keeps real observations; val/test use gt_mask.
            cond = c_mask if mode == "train" else c_gt_mask
            split = make_windows(data_std, cond, c_mask, eval_length, "test" if mode == "test" else "train")

            X_all.append(split["X"])
            X_ori_all.append(split["X_ori"])
            eval_all.append(split["eval_mask"])
            cut_all.append(split["cut_length"])
            start_all.append(split["starts"])
            month_all.append(np.full_like(split["starts"], month, dtype=np.int32))

        return {
            "X": np.concatenate(X_all, axis=0).astype(np.float32),
            "X_ori": np.concatenate(X_ori_all, axis=0).astype(np.float32),
            "eval_mask": np.concatenate(eval_all, axis=0).astype(np.float32),
            "cut_length": np.concatenate(cut_all, axis=0).astype(np.int32),
            "starts": np.concatenate(start_all, axis=0).astype(np.int32),
            "month": np.concatenate(month_all, axis=0).astype(np.int32),
        }

    splits = {
        "train": build_month_split("train"),
        "val": build_month_split("val"),
        "test": build_month_split("test"),
    }

    meta = {
        "dataset": "aqi36",
        "aqi_root": str(aqi_root),
        "ground_path": str(ground_path),
        "missing_path": str(missing_path),
        "meanstd_path": str(meanstd_path),
        "mean": mean,
        "std": std,
        "eval_length": eval_length,
        "target_dim": target_dim,
        "val_len": val_len,
        "mask_sensor": mask_sensor,
        "month_split": {
            "train": [1, 2, 4, 5, 7, 8, 10, 11],
            "valid": [2, 5, 8, 11],
            "test": [3, 6, 9, 12],
        },
    }
    write_dataset_output(out_dir / "aqi36", splits, meta)


# ----------------------------- loader helpers -------------------------------


def load_npz_as_pypots_set(path: str | Path) -> Dict[str, np.ndarray]:
    """Use this in the later training script: model.fit({'X': X_train}, {'X': X_val})."""
    z = np.load(path)
    return {"X": z["X"].astype(np.float32)}


def compute_mae_from_npz(imputation: np.ndarray, eval_npz_path: str | Path, inverse: bool = True) -> float:
    """
    Compute MAE on the user's evaluation mask.

    imputation: model output in standardized scale, shape [N, L, K].
    eval_npz_path: val_set.npz or test_set.npz.
    inverse: if True, report MAE in original units by applying X * std + mean.
             If False, report MAE in standardized scale.
    """
    z = np.load(eval_npz_path)
    y_true = z["X_ori"].astype(np.float32)
    mask = z["eval_mask"].astype(bool)

    y_pred = imputation.astype(np.float32)
    if inverse:
        meta_arrays = np.load(Path(eval_npz_path).with_name("meta_arrays.npz"))
        mean = meta_arrays["mean"].reshape(1, 1, -1)
        std = meta_arrays["std"].reshape(1, 1, -1)
        y_pred = y_pred * std + mean
        y_true = y_true * std + mean

    return float(np.abs(y_pred[mask] - y_true[mask]).mean())


# ----------------------------------- CLI ------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "pems04", "pems08", "aqi36"])
    parser.add_argument("--data-root", type=Path, default=Path("./data"), help="Root containing PEMS04/PEMS08 folders.")
    parser.add_argument("--aqi-root", type=Path, default=Path("/home/duanlei/PriSTI/data/pm25"))
    parser.add_argument("--out-dir", type=Path, default=Path("./pypots_data"))
    parser.add_argument("--missing-pattern", type=str, default="point", choices=["block", "point"])
    parser.add_argument("--missing-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=9101112)
    parser.add_argument("--val-len", type=float, default=0.1)
    parser.add_argument("--test-len", type=float, default=0.2)
    parser.add_argument("--pems-eval-length", type=int, default=24)
    parser.add_argument("--aqi-eval-length", type=int, default=36)
    parser.add_argument("--aqi-target-dim", type=int, default=36)
    parser.add_argument("--mask-sensor", type=int, nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dataset in {"all", "pems04"}:
        build_pems(
            "pems04",
            data_root=args.data_root,
            out_dir=args.out_dir,
            eval_length=args.pems_eval_length,
            val_len=args.val_len,
            test_len=args.test_len,
            missing_pattern=args.missing_pattern,
            missing_ratio=args.missing_ratio,
            seed=args.seed,
        )
    if args.dataset in {"all", "pems08"}:
        build_pems(
            "pems08",
            data_root=args.data_root,
            out_dir=args.out_dir,
            eval_length=args.pems_eval_length,
            val_len=args.val_len,
            test_len=args.test_len,
            missing_pattern=args.missing_pattern,
            missing_ratio=args.missing_ratio,
            seed=args.seed,
        )
    if args.dataset in {"all", "aqi36"}:
        build_aqi36(
            aqi_root=args.aqi_root,
            out_dir=args.out_dir,
            eval_length=args.aqi_eval_length,
            target_dim=args.aqi_target_dim,
            val_len=args.val_len,
            mask_sensor=args.mask_sensor,
        )

    print(f"Done. Converted files are under: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
