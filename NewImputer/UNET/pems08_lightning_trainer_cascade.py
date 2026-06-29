import sys

sys.path.append("/home/duanlei/PriSTI/NewImputer/")

from pems08_lightning_datamodule import Pems08_DataModule
from pems08_lightning_module_cascade import pems08_lightning_module
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import random
import torch
import numpy as np
from transfomer_imputer_pems08_cascade import transformer_imputer


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
loss_flag = "midSupervision"  # "l1" or "threshold" or "midSupervision" or "midEnhance"
lowpass_mode = "fft"  # "avg" or "fft"
fft_keep_ratio = 0.9
aux_lowpass_weight = 0.1
device_num = 3
epochs = 40
device = f"cuda:{device_num}"

pems08_dm = Pems08_DataModule()
model = transformer_imputer(device=device)

if train_flag:
    pems08_imputer = pems08_lightning_module(
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
    pems08_imputer = pems08_lightning_module.load_from_checkpoint(
        checkpoint_path="/home/duanlei/PriSTI/checkpoints/pems08_cascade/last.ckpt",
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
    dirpath="checkpoints/pems08_cascade",
    filename="best_model",
    save_top_k=1,
    save_last=True,
    mode="min",
    verbose=True,
)
logger = TensorBoardLogger(save_dir="logs", name=f"unet_imputer_pems08_cascade_{loss_flag}_lr5e-4")

trainer = Trainer(
    devices=[device_num],
    max_epochs=epochs,
    accelerator="gpu",
    callbacks=[checkpoint_callback],
    deterministic=True,
    limit_train_batches=78 * 10,
    logger=logger,
)

if train_flag:
    trainer.fit(pems08_imputer, pems08_dm)

trainer.test(pems08_imputer, pems08_dm)
