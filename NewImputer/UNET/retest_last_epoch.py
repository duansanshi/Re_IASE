"""Batch retest all v4/v5 experiments using last.ckpt (last epoch) instead of best checkpoint"""
import sys
import os
sys.path.append("/home/duanlei/PriSTI/NewImputer/")
sys.path.append("/home/duanlei/PriSTI/")
import argparse
import random
import torch
import numpy as np
from pytorch_lightning import Trainer

def set_seed(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument("--exp", type=str, required=True, help="experiment key from mapping")
parser.add_argument("--device", type=int, default=0)
args = parser.parse_args()

# Mapping: exp_key -> (dataset, module_version, checkpoint_base, last_ckpt_path)
# checkpoint paths are relative to /home/duanlei/PriSTI/
EXPERIMENTS = {
    # === AQI36 v4 experiments (checkpoints in NewImputer/UNET/checkpoints/) ===
    "aqi_v4_curr03": ("aqi36", "v4", "NewImputer/UNET/checkpoints/myimputer_aqi36_v4_curr0.3_warm0.1"),
    "aqi_v4_s03":    ("aqi36", "v4", "NewImputer/UNET/checkpoints/myimputer_aqi36_v4_curr0.3_smooth0.03"),
    "aqi_v4_s01":    ("aqi36", "v4", "NewImputer/UNET/checkpoints/myimputer_aqi36_v4_curr0.3_smooth0.01"),
    # === AQI36 v5 experiments ===
    "aqi_v5_t1":     ("aqi36", "v5", "NewImputer/UNET/checkpoints/myimputer_aqi36_v5_soft_t1.0_c0.3_s0.03"),
    "aqi_v5_t05":    ("aqi36", "v5", "NewImputer/UNET/checkpoints/myimputer_aqi36_v5_soft_t0.5_c0.3_s0.03"),
    "aqi_v5_t05s01": ("aqi36", "v5", "checkpoints/myimputer_aqi36_aqi36_v5_t05_s01"),
    "aqi_v5_t03s03": ("aqi36", "v5", "checkpoints/myimputer_aqi36_aqi36_v5_t03_s03"),
    "aqi_v5_cw05":   ("aqi36", "v5", "checkpoints/myimputer_aqi36_aqi36_v5_t05_cw05"),
    "aqi_v5_cw02":   ("aqi36", "v5", "checkpoints/myimputer_aqi36_aqi36_v5_t05_cw02"),
    # === PEMS08 v4 experiments ===
    "p08_v4":        ("pems08", "v4", "checkpoints/myimputer_pems08_v4_curr0.3"),
    "p08_v4_s03":    ("pems08", "v4", "checkpoints/myimputer_pems08_v4_curr0.3_smooth0.03"),
    "p08_v4_s01":    ("pems08", "v4", "checkpoints/myimputer_pems08_v4_curr0.3_smooth0.01"),
    "p08_v4_s005":   ("pems08", "v4", "checkpoints/myimputer_p08_v4_s005"),
    # === PEMS08 v5 experiments ===
    "p08_v5_t05":    ("pems08", "v5", "checkpoints/myimputer_pems08_v5_t0.5_c0.3_s0.03"),
    "p08_v5_t1":     ("pems08", "v5", "checkpoints/myimputer_pems08_v5_t1.0_c0.3_s0.03"),
    "p08_v5_t05s01": ("pems08", "v5", "checkpoints/myimputer_p08_v5_t05_s01"),
    "p08_v5_t05s005":("pems08", "v5", "checkpoints/myimputer_p08_v5_t05_s005"),
    "p08_v5_t03s01": ("pems08", "v5", "checkpoints/myimputer_p08_v5_t03_s01"),
    # === PEMS04 v4 experiments ===
    "p04_v4":        ("pems04", "v4", "checkpoints/myimputer_pems04_v4_curr0.3"),
    "p04_v4_s03":    ("pems04", "v4", "checkpoints/myimputer_pems04_v4_curr0.3_smooth0.03"),
    "p04_v4_s01":    ("pems04", "v4", "checkpoints/myimputer_pems04_v4_curr0.3_smooth0.01"),
    "p04_v4_cw05":   ("pems04", "v4", "checkpoints/myimputer_p04_v4_cw05"),
    # === PEMS04 v5 experiments ===
    "p04_v5_t05":    ("pems04", "v5", "checkpoints/myimputer_pems04_v5_t0.5_c0.3_s0.03"),
    "p04_v5_t05s0":  ("pems04", "v5", "checkpoints/myimputer_p04_v5_t05_s0"),
}

if args.exp not in EXPERIMENTS:
    print(f"Unknown experiment: {args.exp}")
    print(f"Available: {list(EXPERIMENTS.keys())}")
    sys.exit(1)

dataset, version, ckpt_base = EXPERIMENTS[args.exp]
ckpt_path = os.path.join("/home/duanlei/PriSTI", ckpt_base, "last.ckpt")

if not os.path.exists(ckpt_path):
    print(f"ERROR: Checkpoint not found: {ckpt_path}")
    sys.exit(1)

set_seed(2026)
device = f"cuda:{args.device}"

if dataset == "aqi36":
    from aqi36_lightning_datamodule import AQI36_DataModule
    from transformer_imputer_new import transformer_imputer
    dm = AQI36_DataModule(device=device)
    model = transformer_imputer(device=device)
    if version == "v4":
        from imputer_lightning_module_new_v4 import aqi_lightning_module
    else:
        from imputer_lightning_module_new_v5 import aqi_lightning_module
    imputer = aqi_lightning_module.load_from_checkpoint(checkpoint_path=ckpt_path, model=model, lr=5e-4)

elif dataset == "pems08":
    from pems08_lightning_datamodule import Pems08_DataModule
    from transfomer_imputer_pems08 import transformer_imputer
    dm = Pems08_DataModule()
    model = transformer_imputer(device=device)
    if version == "v4":
        from pems08_lightning_module_v4 import pems08_lightning_module
    else:
        from pems08_lightning_module_v5 import pems08_lightning_module
    imputer = pems08_lightning_module.load_from_checkpoint(checkpoint_path=ckpt_path, model=model, lr=5e-4)

elif dataset == "pems04":
    from pems04_lightning_datamodule import Pems04_DataModule
    from transformer_imputer_pems04 import transformer_imputer
    dm = Pems04_DataModule()
    model = transformer_imputer(device=device)
    if version == "v4":
        from pems04_lightning_module_v4 import pems04_lightning_module
    else:
        from pems04_lightning_module_v5 import pems04_lightning_module
    imputer = pems04_lightning_module.load_from_checkpoint(checkpoint_path=ckpt_path, model=model, lr=5e-4)

print(f"\n=== Retesting {args.exp} ({dataset} {version}) with LAST EPOCH checkpoint ===")
print(f"Checkpoint: {ckpt_path}\n")

trainer = Trainer(
    devices=[args.device],
    accelerator="gpu",
    deterministic=True,
)
trainer.test(imputer, dm)
