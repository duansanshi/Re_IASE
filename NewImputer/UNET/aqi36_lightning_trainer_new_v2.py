from aqi36_lightning_datamodule import AQI36_DataModule
from imputer_lightning_module_new_v2 import aqi_lightning_module
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

#############################
# 命令行参数
#############################
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--loss_flag", type=str, default="l1", help="l1 | midSupervision | midEnhance")
parser.add_argument("--smooth_weight", type=float, default=0.05)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--decay", action="store_true", default=False)
parser.add_argument("--test_only", action="store_true", default=False)
parser.add_argument("--no_test", action="store_true", default=False, help="skip test after training")
parser.add_argument("--checkpoint_path", type=str, default=None)
parser.add_argument("--exp_name", type=str, default=None)
args = parser.parse_args()

#############################
# 实验配置
#############################
train_flag = not args.test_only
test_flag = not args.no_test
device_num = args.device
epochs = args.epochs
device = f"cuda:{device_num}"

aqi36_dm = AQI36_DataModule(device=device)
model = transformer_imputer(device=device)

# 自动生成实验名称
if args.exp_name:
    exp_name = args.exp_name
else:
    exp_name = f"new_v2_{args.loss_flag}"
    if args.smooth_weight > 0:
        exp_name += f"_smooth{args.smooth_weight}"

print(f"\n{'='*50}")
print(f"Experiment: {exp_name}")
print(f"Device: cuda:{device_num}, Epochs: {epochs}")
print(f"Loss: {args.loss_flag}, Smooth: {args.smooth_weight}, LR: {args.lr}")
print(f"{'='*50}\n")

if train_flag:
    aqi36_imputer = aqi_lightning_module(
        model, lr=args.lr, loss_flag=args.loss_flag, epochs=epochs,
        decay_flag=args.decay, smooth_weight=args.smooth_weight,
    )
else:
    if args.checkpoint_path is None:
        raise ValueError("当 --test_only 时，必须指定 --checkpoint_path 参数")
    print(f"Loading model from checkpoint: {args.checkpoint_path}")
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
    verbose=True
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
        print("\nTraining completed. Test skipped.")
else:
    if test_flag:
        print("\n" + "="*50)
        print("Running Test...")
        print("="*50)
        trainer.test(aqi36_imputer, aqi36_dm)
