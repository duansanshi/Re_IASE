import pytorch_lightning as pl
from dataset_aqi36 import get_dataloader


class AQI36_DataModule(pl.LightningDataModule):
    def __init__(self,batch_size=16,device="cuda:0",val_len=0.1,
                 is_interpolate=True,num_workers=16,target_strategy="hybrid",mask_sensor=[]):
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.val_len = val_len
        self.is_interpolate = is_interpolate
        self.num_workers = num_workers
        self.target_strategy = target_strategy
        self.mask_sensor = mask_sensor

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        """
        Setup method called on every device to split and preprocess data
        """
        self.train_loader,self.valid_loader,self.test_loader,self.scaler,self.mean_scaler=get_dataloader(
            batch_size=self.batch_size,
            device=self.device,
            val_len=self.val_len,
            is_interpolate=self.is_interpolate,
            num_workers=self.num_workers,
            target_strategy=self.target_strategy,
            mask_sensor=self.mask_sensor
        )

    def train_dataloader(self):
        """
        Returns the train dataloader
        """
        return self.train_loader

    def val_dataloader(self):
        """
        Returns the validation dataloader
        """
        return self.valid_loader

    def test_dataloader(self):
        """
        Returns the test dataloader
        """
        return self.test_loader


