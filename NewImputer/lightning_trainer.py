from aqi36_lightning_datamodule import AQI36_DataModule
from lightning_module import AQI_LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint
import torch.nn as nn
from pristi import Guide_diff
from csdi import diff_CSDI

import yaml
flag_list = ["pristi","csdi"]
flag = flag_list[1]

device_num = 2
device = f"cuda:{device_num}"
##################################
if flag == "pristi":
    path = "/home/duanlei/PriSTI/PriSTI/config/aqi.yaml" 
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    config["model"]["is_unconditional"] = False
    config["model"]["target_strategy"] = "hybrid"
    config["diffusion"]["adj_file"] = 'AQI36'
    config["seed"] = 42
    config_diff = config["diffusion"]
    config_diff["side_dim"] = config["model"]["timeemb"]+config["model"]["featureemb"]
    config_diff["device"] = device
    input_dim = 2
##################################
aqi36_dm = AQI36_DataModule(device=device)
if flag == "pristi":
    model = Guide_diff(config_diff, input_dim, target_dim=36, is_itp=True)
else:
    #model = diff_CSDI(device=device, inputdim=2)
    model = diff_CSDI(device=device, inputdim=3)
train_flag = True

if train_flag:
    aqi36_imputer = AQI_LightningModule(model,id,lr=1e-3)
else:
    aqi36_imputer = AQI_LightningModule.load_from_checkpoint(checkpoint_path="/home/duanlei/PriSTI/checkpoints/best_model-v67.ckpt",model=model,id=id,lr=1e-4)


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
    trainer.fit(aqi36_imputer,aqi36_dm)
trainer.test(aqi36_imputer,aqi36_dm)