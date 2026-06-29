from aqi36_lightning_datamodule import AQI36_DataModule
from best_tmp_lightning import aqi_lightning_module
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint
import torch.nn as nn
from best_tmp_imputer import transformer_imputer
#from transformer_imputer import TimeSeriesUNet

import yaml


train_flag = True
device_num = 5
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
    aqi36_imputer = aqi_lightning_module(model,lr=5e-4)
else:
    aqi36_imputer = aqi_lightning_module.load_from_checkpoint(checkpoint_path="/home/duanlei/PriSTI/checkpoints/best_model.ckpt",model=model,lr=1e-4)


early_stop_callback = EarlyStopping(
    monitor="val_loss",
    patience=20,
    verbose=True,
    mode="min",
    check_on_train_epoch_end=False
)
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="checkpoints",
    filename="best_model",
    save_top_k=1,
    mode="min",
    verbose=True
)
trainer = Trainer(
    #accelerator="auto",
    devices = [device_num],
    max_epochs = 200,
    callbacks=[early_stop_callback,checkpoint_callback]
    #val_check_interval=5
)
if train_flag:
    trainer.fit(aqi36_imputer, aqi36_dm)
    best_model_path = checkpoint_callback.best_model_path
    aqi36_imputer = aqi_lightning_module.load_from_checkpoint(best_model_path,model=model, lr=1e-4)
trainer.test(aqi36_imputer, aqi36_dm)