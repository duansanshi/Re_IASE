from aqi36_lightning_datamodule import AQI36_DataModule
from imputer_lightning_module_new import aqi_lightning_module
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import random
import torch
import numpy as np
import torch.nn as nn
from transformer_imputer_new import transformer_imputer
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
loss_flag = "l1"
device_num = 2
epochs = 200
device = f"cuda:{device_num}"
##################################

##################################
aqi36_dm = AQI36_DataModule(device=device)
#model = transformer_imputer(device=device,d_model=128)
model = transformer_imputer(device=device)
#model = TimeSeriesUNet(
#     device=device,input_dim=36
# )

if train_flag:
    aqi36_imputer = aqi_lightning_module(model,lr=5e-4,loss_flag=loss_flag,epochs=epochs,decay_flag=Decay)
else:
    aqi36_imputer = aqi_lightning_module.load_from_checkpoint(checkpoint_path="/home/duanlei/PriSTI/checkpoints/mae_9.21.ckpt",model=model,lr=1e-4)


early_stop_callback = EarlyStopping(
    monitor="val_loss",
    patience=20,
    verbose=True,
    mode="min",
    check_on_train_epoch_end=False
)
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="checkpoints/baseline_aqi36_new",
    filename="best_model",
    save_top_k=1,
    mode="min",
    verbose=True
)
logger = TensorBoardLogger(save_dir="logs", name=f"unet_imputer_aqi36_{loss_flag}")
trainer = Trainer(
    devices = [device_num],
    max_epochs = epochs,
    callbacks=[checkpoint_callback],
    deterministic=True,
    logger=logger,
)
if train_flag:
    trainer.fit(aqi36_imputer, aqi36_dm)
trainer.test(aqi36_imputer, aqi36_dm)