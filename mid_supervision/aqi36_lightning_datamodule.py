import sys
sys.path.append("/home/duanlei/PriSTI")
from pytorch_lightning import LightningDataModule
from dataset_aqi36 import AQI36_Dataset
from torch.utils.data import DataLoader
import torch
class AQI36DataModule(LightningDataModule):
    def __init__(self, batch_size=64, device='cuda:0', val_len=0.1, is_interpolate=True,
                 num_workers=16, target_strategy='hybrid', mask_sensor=None):
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.val_len = val_len
        self.is_interpolate = is_interpolate
        self.num_workers = num_workers
        self.target_strategy = target_strategy
        self.mask_sensor = mask_sensor if mask_sensor is not None else []

    def setup(self, stage=None):
        self.train_dataset = AQI36_Dataset(mode="train", is_interpolate=self.is_interpolate,
                                           target_strategy=self.target_strategy, mask_sensor=self.mask_sensor)
        self.valid_dataset = AQI36_Dataset(mode="valid", val_len=self.val_len,
                                           is_interpolate=self.is_interpolate, target_strategy=self.target_strategy, mask_sensor=self.mask_sensor)
        self.test_dataset = AQI36_Dataset(mode="test", is_interpolate=self.is_interpolate,
                                          target_strategy=self.target_strategy, mask_sensor=self.mask_sensor)

        self.scaler = torch.from_numpy(self.train_dataset.train_std).to(self.device).float()
        self.mean_scaler = torch.from_numpy(self.train_dataset.train_mean).to(self.device).float()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)