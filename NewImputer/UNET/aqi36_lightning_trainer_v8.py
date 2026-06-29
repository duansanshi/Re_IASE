"""
v8 Trainer: 频域损失正则化
使用 v5 架构 (transformer_imputer_new) + v8 lightning module (频域损失)
"""
from aqi36_lightning_datamodule import AQI36_DataModule
from imputer_lightning_module_v8 import aqi_lightning_module
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import random
import torch
import numpy as np
import argparse
from transformer_imputer_new import transformer_imputer


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
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--curriculum_weight", type=float, default=0.3)
parser.add_argument("--curriculum_warmup_ratio", type=float, default=0.1)
parser.add_argument("--curriculum_temperature", type=float, default=1.0)
parser.add_argument("--smooth_weight", type=float, default=0.03)
parser.add_argument("--freq_loss_weight", type=float, default=0.0,
                    help="频域损失权重。0=禁用。建议范围 0.01~0.5")
parser.add_argument("--preimpute", type=str, default="Forward")
parser.add_argument("--no_freq", action="store_true")
parser.add_argument("--freq_ratio", type=float, default=0.1)
parser.add_argument("--mavg_window", type=int, default=0)
parser.add_argument("--decay_alpha", type=float, default=0.0)
parser.add_argument("--dct_ratio", type=float, default=0.0)
parser.add_argument("--wavelet_level", type=int, default=0)
parser.add_argument("--test_only", action="store_true", default=False)
parser.add_argument("--no_test", action="store_true", default=False)
parser.add_argument("--checkpoint_path", type=str, default=None)
parser.add_argument("--exp_name", type=str, default=None)
args = parser.parse_args()

train_flag = not args.test_only
test_flag = not args.no_test
device_num = args.device
epochs = args.epochs
device = f"cuda:{device_num}"

aqi36_dm = AQI36_DataModule(
    device=device, preimpute_flag=args.preimpute,
    freq_flag=not args.no_freq, freq_ratio=args.freq_ratio,
    mavg_window=args.mavg_window, decay_alpha=args.decay_alpha,
    dct_ratio=args.dct_ratio, wavelet_level=args.wavelet_level
)
model = transformer_imputer(device=device)

if args.exp_name:
    exp_name = args.exp_name
else:
    exp_name = f"v8_floss{args.freq_loss_weight}_t{args.curriculum_temperature}"

print(f"\n{'='*50}")
print(f"Experiment: {exp_name}")
print(f"Device: cuda:{device_num}, Epochs: {epochs}, LR: {args.lr}")
print(f"Curriculum: weight={args.curriculum_weight}, temp={args.curriculum_temperature}")
print(f"Smooth: {args.smooth_weight}, Freq loss: {args.freq_loss_weight}")
print(f"{'='*50}\n")

if train_flag:
    aqi36_imputer = aqi_lightning_module(
        model, lr=args.lr, epochs=epochs,
        smooth_weight=args.smooth_weight,
        curriculum_weight=args.curriculum_weight,
        curriculum_warmup_ratio=args.curriculum_warmup_ratio,
        curriculum_temperature=args.curriculum_temperature,
        freq_loss_weight=args.freq_loss_weight,
    )
else:
    if args.checkpoint_path is None:
        raise ValueError("当 --test_only 时，必须指定 --checkpoint_path")
    aqi36_imputer = aqi_lightning_module.load_from_checkpoint(
        checkpoint_path=args.checkpoint_path, model=model, lr=args.lr
    )

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=f"checkpoints/myimputer_aqi36_{exp_name}",
    filename="best_model",
    save_top_k=1,
    save_last=True,
    mode="min",
    verbose=True,
)

logger = TensorBoardLogger(save_dir="logs", name=f"unet_imputer_aqi36_{exp_name}")

trainer = Trainer(
    devices=[device_num],
    max_epochs=epochs,
    callbacks=[checkpoint_callback],
    deterministic=True,
    logger=logger,
)

if train_flag:
    trainer.fit(aqi36_imputer, aqi36_dm)
    if test_flag:
        best_model_path = checkpoint_callback.best_model_path
        print(f"\nLoading best model from {best_model_path}")
        aqi36_imputer = aqi_lightning_module.load_from_checkpoint(
            checkpoint_path=best_model_path, model=model, lr=args.lr
        )
        print("\n" + "="*50)
        print("Running Test...")
        print("="*50)
        trainer.test(aqi36_imputer, aqi36_dm)
else:
    if test_flag:
        print("\n" + "="*50)
        print("Running Test...")
        print("="*50)
        trainer.test(aqi36_imputer, aqi36_dm)
