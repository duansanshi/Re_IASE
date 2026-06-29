from dataset_pemsbay import get_dataloader
import pytorch_lightning as pl
from main_model import PriSTI_PemsBAY

import numpy as np
import torch
import yaml
import json
import datetime

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


with open("/home/duanlei/PriSTI/config/traffic.yaml", "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = 0
config["model"]["target_strategy"] = "hybrid"
config["diffusion"]["adj_file"] = 'pems-bay'
config["seed"] = SEED




model = PriSTI_PemsBAY(config, "cuda:0").to("cuda:0")
train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
        batch_size=4, device="cuda:0", missing_pattern='block',
        is_interpolate=True, num_workers=4,
        target_strategy='hybrid',
    )

print(len(train_loader))

class PemsBayDataModule(pl.LightningDataModule):
    def train_dataloader(self):
        return train_loader
    
    def val_dataloader(self):
        return valid_loader
    
    def test_dataloader(self):
        return test_loader
    


if __name__ == "__main__":
    pass
  