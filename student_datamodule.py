from dataset_aqi36 import get_dataloader
import pytorch_lightning as pl
from main_model import PriSTI_aqi36

import numpy as np
import torch
import yaml
import json
import datetime

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

with open("/home/duanlei/PriSTI/config/base.yaml", "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = 0
config["model"]["target_strategy"] = "hybrid"
config["diffusion"]["adj_file"] = 'AQI36'
config["seed"] = SEED




#model = PriSTI_aqi36(config, "cuda:0").to("cuda:0")
train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
        batch_size=16, device="cuda:0", val_len=0.1,
        is_interpolate=True, num_workers=16,
        target_strategy="hybrid", mask_sensor=[]
    )
# cnt = 0
# for batch in test_loader:
#         (
#             observed_data,
#             observed_mask,
#             observed_tp,
#             gt_mask,
#             _,
#             cut_length,
#             coeffs,
#             _,
#         ) = model.process_data(batch)
#         eval_mask = observed_mask-gt_mask
#         cnt+=eval_mask.sum().item()
# print(cnt)

class StudentDataModule(pl.LightningDataModule):
    def train_dataloader(self):
        return train_loader
    
    def val_dataloader(self):
        return valid_loader
    
    def test_dataloader(self):
        return test_loader
    


if __name__ == "__main__":
    dm = StudentDataModule()
    train_loader = dm.train_dataloader()
    for batch in train_loader:
        this = batch
        break
    print(this)
  