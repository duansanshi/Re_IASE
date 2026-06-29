from aqi36_lightning_datamodule import AQI36_DataModule
from lightning_module import AQI_LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint
import torch.nn as nn
from pristi import Guide_diff
from myimputer import transformer_imputer

import yaml
flag_list = ["pristi","myimputer"]
flag = flag_list[1]
n_samples_list = [1,5,10,20,50]
device_num = 2
epochs = 300
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
    model = transformer_imputer(device=device)
train_flag = False

if train_flag:
    aqi36_imputer = AQI_LightningModule(model,id,lr=1e-4)
else:
    aqi36_imputer = AQI_LightningModule.load_from_checkpoint(checkpoint_path="/home/duanlei/PriSTI/checkpoints/rectified_flow_mae_myflow/best_model_400.ckpt",model=model,id=id,lr=1e-4)
aqi36_imputer.flag = flag

early_stop_callback = EarlyStopping(
    monitor="val_loss",
    patience=20,
    verbose=True,
    mode="min",
    check_on_train_epoch_end=False
)
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="/home/duanlei/PriSTI/checkpoints/rectified_flow_mae_test",
    filename=f"best_model_{epochs}",
    save_last= True,
    mode="min",
    verbose=True
)
trainer = Trainer(
    #accelerator="auto",
    devices = [device_num],
    max_epochs = epochs,
    callbacks=[checkpoint_callback]
    #val_check_interval=5
)
if train_flag:
    trainer.fit(aqi36_imputer,aqi36_dm)
for n_samples in n_samples_list:
    aqi36_imputer.n_samples = n_samples
    trainer.test(aqi36_imputer,aqi36_dm)
    with open("results.txt", "a") as f:
        f.write(f"n_samples = {n_samples}, MAE = {aqi36_imputer.MAE}\n")

