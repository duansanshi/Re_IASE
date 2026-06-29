import sys
import os
import argparse

sys.path.append("/home/duanlei/PriSTI/NewImputer/")
sys.path.append("/home/duanlei/PriSTI/")

from pems04_lightning_datamodule import Pems04_DataModule
from pems04_lightning_module import pems04_lightning_module
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import random
import torch
import numpy as np
from transformer_imputer_pems04 import transformer_imputer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 2026
set_seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--preimpute", type=str, default="Linear")
parser.add_argument("--freq", action="store_true")
parser.add_argument("--no_freq", action="store_true")
parser.add_argument("--exp_name", type=str, default="p04_ablation")
args = parser.parse_args()

freq_flag = not args.no_freq
device_num = args.device
epochs = 40
device = f"cuda:{device_num}"

pems04_dm = Pems04_DataModule(preimpute_flag=args.preimpute, freq_flag=freq_flag)
model = transformer_imputer(device=device)
imputer = pems04_lightning_module(model, lr=5e-4, loss_flag="l1", epochs=epochs, decay_flag=False)

ckpt_dir = f"checkpoints/{args.exp_name}"
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=ckpt_dir,
    filename="best_model",
    save_top_k=1,
    mode="min",
    verbose=True,
    save_last=True,
)

trainer = Trainer(
    devices=[device_num],
    max_epochs=epochs,
    accelerator="gpu",
    callbacks=[checkpoint_callback],
    deterministic=True,
    limit_train_batches=742,
)

trainer.fit(imputer, pems04_dm)
best_model_path = checkpoint_callback.best_model_path
print(f"\nLoading best model from {best_model_path}")
imputer = pems04_lightning_module.load_from_checkpoint(best_model_path, model=model, lr=5e-4)
trainer.test(imputer, pems04_dm)
