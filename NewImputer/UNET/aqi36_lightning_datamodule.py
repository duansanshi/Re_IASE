import pytorch_lightning as pl
from dataset_aqi import get_dataloader


class AQI36_DataModule(pl.LightningDataModule):
    def __init__(self,batch_size=16,device="cuda:0",val_len=0.1,
                 is_interpolate=True,num_workers=16,target_strategy="hybrid",mask_sensor=[],preimpute_flag="Linear",freq_flag=True,
                 freq_ratio=0.1, mavg_window=0, decay_alpha=0.0, dct_ratio=0.0, wavelet_level=0,
                 aux_freq=False, aux_freq_ratio=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.val_len = val_len
        self.is_interpolate = is_interpolate
        self.num_workers = num_workers
        self.target_strategy = target_strategy
        self.mask_sensor = mask_sensor
        self.preimpute_flag = preimpute_flag
        self.freq_flag = freq_flag
        self.freq_ratio = freq_ratio
        self.mavg_window = mavg_window
        self.decay_alpha = decay_alpha
        self.dct_ratio = dct_ratio
        self.wavelet_level = wavelet_level
        self.aux_freq = aux_freq
        self.aux_freq_ratio = aux_freq_ratio

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
            mask_sensor=self.mask_sensor,
            preimpute_flag=self.preimpute_flag,
            freq_flag=self.freq_flag,
            freq_ratio=self.freq_ratio,
            mavg_window=self.mavg_window,
            decay_alpha=self.decay_alpha,
            dct_ratio=self.dct_ratio,
            wavelet_level=self.wavelet_level,
            aux_freq=self.aux_freq,
            aux_freq_ratio=self.aux_freq_ratio
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


