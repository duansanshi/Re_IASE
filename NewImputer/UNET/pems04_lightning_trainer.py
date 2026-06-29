import sys
import os 

# 将父目录添加到 sys.path 中
sys.path.append("/home/duanlei/PriSTI/NewImputer/")
sys.path.append("/home/duanlei/PriSTI/")

from pems04_lightning_datamodule import Pems04_DataModule
from pems04_lightning_module import pems04_lightning_module
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint
import random
import torch
import numpy as np
import torch.nn as nn
from transformer_imputer_pems04 import transformer_imputer
#from transformer_imputer import TimeSeriesUNet

import yaml

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置随机种子值
SEED = 2026  # 你可以选择任何你喜欢的数字
set_seed(SEED)



train_flag = True
Decay = False
loss_flag = "l1" # "l1" or "threshold" or "midSupervision"
device_num = 0
epochs = 40
device = f"cuda:{device_num}"
##################################

##################################
pems04_dm = Pems04_DataModule()
#model = transformer_imputer(device=device,d_model=128)
model = transformer_imputer(device=device)
#model = TimeSeriesUNet(
#     device=device,input_dim=307
# )

if train_flag:
    aqi36_imputer = pems04_lightning_module(model,lr=5e-4,loss_flag=loss_flag,epochs=epochs,decay_flag=Decay)
else:
    aqi36_imputer = pems04_lightning_module.load_from_checkpoint(checkpoint_path="/home/duanlei/PriSTI/checkpoints/best_model-v75.ckpt",model=model,lr=1e-4)


early_stop_callback = EarlyStopping(
    monitor="val_loss",
    patience=20,
    verbose=True,
    mode="min",
    check_on_train_epoch_end=False
)
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="checkpoints/baseline_pems04_freq",
    filename="best_model",
    save_top_k=1,
    mode="min",
    verbose=True,
    save_last=True,
)
# trainer = Trainer(
#     #accelerator="auto",
#     devices = [device_num],
#     max_epochs = 200,
#     # callbacks=[early_stop_callback,checkpoint_callback],
#     callbacks=[checkpoint_callback],
#     deterministic=True,
#     #val_check_interval=5
# )
trainer = Trainer(
    devices = [device_num],
    max_epochs=epochs,
    accelerator="gpu",
    callbacks=[checkpoint_callback],
    deterministic=True,
    limit_train_batches=742, 
)
if train_flag:
    trainer.fit(aqi36_imputer, pems04_dm)
    best_model_path = checkpoint_callback.best_model_path
    print(f"\nLoading best model from {best_model_path}")
    aqi36_imputer = pems04_lightning_module.load_from_checkpoint(best_model_path, model=model, lr=5e-4)
trainer.test(aqi36_imputer, pems04_dm)