from aqi36_lightning_datamodule import AQI36_DataModule
from imputer_lightning_module_cascade_v2 import aqi_lightning_module
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import random
import torch
import numpy as np
import argparse
from transformer_imputer_cascade import transformer_imputer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = 2026
set_seed(SEED)

#############################
# 命令行参数
#############################
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--loss_flag", type=str, default="midEnhance")
parser.add_argument("--smooth_weight", type=float, default=0.05)
parser.add_argument("--dist_aware", action="store_true", default=False)
parser.add_argument("--dist_mode", type=str, default="sqrt")
parser.add_argument("--dist_temperature", type=float, default=2.0)
parser.add_argument("--dist_layer_aware", action="store_true", default=False)
parser.add_argument("--aux_lowpass_weight", type=float, default=0.0)
parser.add_argument("--lowpass_mode", type=str, default="avg")
parser.add_argument("--fft_keep_ratio", type=float, default=0.9)
parser.add_argument("--test_only", action="store_true", default=False)
parser.add_argument("--checkpoint_path", type=str, default=None)
parser.add_argument("--exp_name", type=str, default=None)
args = parser.parse_args()

#############################
# 实验配置
#############################
train_flag = not args.test_only
test_flag = True
device_num = args.device
epochs = args.epochs
device = f"cuda:{device_num}"

#############################
# 构建模型和 lightning module
#############################
aqi36_dm = AQI36_DataModule(device=device)
model = transformer_imputer(device=device)

# 自动生成实验名称
if args.exp_name:
    exp_name = args.exp_name
else:
    exp_name = f"cascade_v2_{args.loss_flag}"
    if args.dist_aware:
        exp_name += f"_dist{args.dist_mode}"
        if args.dist_layer_aware:
            exp_name += "_layered"
    if args.smooth_weight > 0:
        exp_name += f"_smooth{args.smooth_weight}"

print(f"\n{'='*50}")
print(f"Experiment: {exp_name}")
print(f"Device: cuda:{device_num}, Epochs: {epochs}")
print(f"Loss: {args.loss_flag}, Smooth: {args.smooth_weight}, DistAware: {args.dist_aware}")
print(f"{'='*50}\n")

if train_flag:
    aqi36_imputer = aqi_lightning_module(
        model,
        lr=5e-4,
        loss_flag=args.loss_flag,
        epochs=epochs,
        decay_flag=False,
        aux_lowpass_weight=args.aux_lowpass_weight,
        lowpass_mode=args.lowpass_mode,
        fft_keep_ratio=args.fft_keep_ratio,
        smooth_weight=args.smooth_weight,
        dist_aware=args.dist_aware,
        dist_mode=args.dist_mode,
        dist_temperature=args.dist_temperature,
        dist_layer_aware=args.dist_layer_aware,
    )
else:
    if args.checkpoint_path is None:
        raise ValueError("当 --test_only 时，必须指定 --checkpoint_path 参数")
    print(f"Loading model from checkpoint: {args.checkpoint_path}")
    aqi36_imputer = aqi_lightning_module.load_from_checkpoint(
        checkpoint_path=args.checkpoint_path, model=model, lr=5e-4
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
            checkpoint_path=best_model_path, model=model, lr=5e-4
        )
        print("\n" + "="*50)
        print("Running Test...")
        print("="*50)
        trainer.test(aqi36_imputer, aqi36_dm)
    else:
        print("\nTraining completed. Test skipped (test_flag=False)")
else:
    if test_flag:
        print("\n" + "="*50)
        print("Running Test...")
        print("="*50)
        trainer.test(aqi36_imputer, aqi36_dm)
