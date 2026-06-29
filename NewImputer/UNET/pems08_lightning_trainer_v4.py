import sys
sys.path.append("/home/duanlei/PriSTI/NewImputer/")
sys.path.append("/home/duanlei/PriSTI/")

from pems08_lightning_datamodule import Pems08_DataModule
from pems08_lightning_module_v4 import pems08_lightning_module
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import random
import torch
import numpy as np
import argparse
from transfomer_imputer_pems08 import transformer_imputer


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
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--curriculum_weight", type=float, default=0.3)
parser.add_argument("--curriculum_warmup_ratio", type=float, default=0.1)
parser.add_argument("--smooth_weight", type=float, default=0.0)
parser.add_argument("--limit_train_batches", type=int, default=780, help="78*10=780")
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

pems08_dm = Pems08_DataModule()
model = transformer_imputer(device=device)

if args.exp_name:
    exp_name = args.exp_name
else:
    exp_name = f"pems08_v4_curr{args.curriculum_weight}"
    if args.smooth_weight > 0:
        exp_name += f"_smooth{args.smooth_weight}"

print(f"\n{'='*50}")
print(f"Experiment: {exp_name}")
print(f"Device: cuda:{device_num}, Epochs: {epochs}, LR: {args.lr}")
print(f"Curriculum: weight={args.curriculum_weight}, warmup={args.curriculum_warmup_ratio}")
print(f"Smooth: {args.smooth_weight}")
print(f"limit_train_batches: {args.limit_train_batches}")
print(f"{'='*50}\n")

if train_flag:
    pems08_imputer = pems08_lightning_module(
        model, lr=args.lr, epochs=epochs,
        smooth_weight=args.smooth_weight,
        curriculum_weight=args.curriculum_weight,
        curriculum_warmup_ratio=args.curriculum_warmup_ratio,
    )
else:
    if args.checkpoint_path is None:
        raise ValueError("当 --test_only 时，必须指定 --checkpoint_path")
    pems08_imputer = pems08_lightning_module.load_from_checkpoint(
        checkpoint_path=args.checkpoint_path, model=model, lr=args.lr
    )

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=f"checkpoints/myimputer_{exp_name}",
    filename="best_model",
    save_top_k=1,
    save_last=True,
    mode="min",
    verbose=True,
)

logger = TensorBoardLogger(save_dir="logs", name=f"unet_imputer_{exp_name}")

trainer = Trainer(
    devices=[device_num],
    max_epochs=epochs,
    accelerator="gpu",
    callbacks=[checkpoint_callback],
    deterministic=True,
    logger=logger,
    limit_train_batches=args.limit_train_batches,
)

if train_flag:
    trainer.fit(pems08_imputer, pems08_dm)
    if test_flag:
        best_model_path = checkpoint_callback.best_model_path
        print(f"\nLoading best model from {best_model_path}")
        pems08_imputer = pems08_lightning_module.load_from_checkpoint(
            checkpoint_path=best_model_path, model=model, lr=args.lr
        )
        print("\n" + "="*50)
        print("Running Test...")
        print("="*50)
        trainer.test(pems08_imputer, pems08_dm)
else:
    if test_flag:
        print("\n" + "="*50)
        print("Running Test...")
        print("="*50)
        trainer.test(pems08_imputer, pems08_dm)
