"""Retest baseline models using best checkpoint (not last epoch)"""
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
parser.add_argument("--dataset", type=str, required=True, choices=["aqi36", "pems08", "pems04"])
parser.add_argument("--device", type=int, default=0)
args = parser.parse_args()

set_seed(2026)
device = f"cuda:{args.device}"

if args.dataset == "aqi36":
    from aqi36_lightning_datamodule import AQI36_DataModule
    from imputer_lightning_module_new import aqi_lightning_module
    from transformer_imputer_new import transformer_imputer
    dm = AQI36_DataModule(device=device)
    model = transformer_imputer(device=device)
    ckpt_path = "/home/duanlei/PriSTI/checkpoints/baseline_aqi36_new/best_model.ckpt"
    imputer = aqi_lightning_module.load_from_checkpoint(checkpoint_path=ckpt_path, model=model, lr=5e-4)

elif args.dataset == "pems08":
    from pems08_lightning_datamodule import Pems08_DataModule
    from pems08_lightning_module import pems08_lightning_module
    from transfomer_imputer_pems08 import transformer_imputer
    dm = Pems08_DataModule()
    model = transformer_imputer(device=device)
    ckpt_path = "/home/duanlei/PriSTI/checkpoints/baseline_pems08/best_model.ckpt"
    imputer = pems08_lightning_module.load_from_checkpoint(checkpoint_path=ckpt_path, model=model, lr=5e-4)

elif args.dataset == "pems04":
    from pems04_lightning_datamodule import Pems04_DataModule
    from pems04_lightning_module import pems04_lightning_module
    from transformer_imputer_pems04 import transformer_imputer
    dm = Pems04_DataModule()
    model = transformer_imputer(device=device)
    ckpt_path = "/home/duanlei/PriSTI/checkpoints/baseline_pems04/best_model.ckpt"
    imputer = pems04_lightning_module.load_from_checkpoint(checkpoint_path=ckpt_path, model=model, lr=5e-4)

print(f"\n=== Retesting {args.dataset} baseline with best checkpoint: {ckpt_path} ===\n")

trainer = Trainer(
    devices=[args.device],
    accelerator="gpu",
    deterministic=True,
)
trainer.test(imputer, dm)
