from aqi36_lightning_datamodule import AQI36_DataModule
from imputer_lightning_module_cascade import aqi_lightning_module
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import random
import torch
import numpy as np
from transformer_imputer_cascade import transformer_imputer
import yaml


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = 2026
set_seed(SEED)

train_flag = True
Decay = False
loss_flag = "midEnhance"  # "l1" or "threshold" or "midSupervision" or "midEnhance"
lowpass_mode = "avg"  # "avg" or "fft"
fft_keep_ratio = 0.9
aux_lowpass_weight = 0.1
device_num = 0
epochs = 200
device = f"cuda:{device_num}"


aqi36_dm = AQI36_DataModule(device=device)
model = transformer_imputer(device=device)

if train_flag:
    aqi36_imputer = aqi_lightning_module(
        model,
        lr=5e-4,
        loss_flag=loss_flag,
        epochs=epochs,
        decay_flag=Decay,
        aux_lowpass_weight=aux_lowpass_weight,
        lowpass_mode=lowpass_mode,
        fft_keep_ratio=fft_keep_ratio,
    )
else:
    aqi36_imputer = aqi_lightning_module.load_from_checkpoint(
        checkpoint_path="/home/duanlei/PriSTI/checkpoints/myimputer_aqi36/last-v12.ckpt",
        model=model,
        lr=1e-4,
    )


early_stop_callback = EarlyStopping(
    monitor="val_loss",
    patience=20,
    verbose=True,
    mode="min",
    check_on_train_epoch_end=False,
)
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="checkpoints/myimputer_aqi36_cascade",
    filename="best_model",
    save_top_k=1,
    save_last=True,
    mode="min",
    verbose=True,
)
logger = TensorBoardLogger(save_dir="logs", name=f"unet_imputer_aqi36_cascade_{loss_flag}_lr1e-4")
trainer = Trainer(
    devices=[device_num],
    max_epochs=epochs,
    callbacks=[checkpoint_callback],
    deterministic=True,
    logger=logger,
)

if train_flag:
    trainer.fit(aqi36_imputer, aqi36_dm)

trainer.test(aqi36_imputer, aqi36_dm)
