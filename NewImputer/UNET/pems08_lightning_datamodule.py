import sys
sys.path.append("./")

from dataset_pems08 import get_dataloader
import pytorch_lightning as pl
import numpy as np
import torch
import yaml

SEED = 42
pl.seed_everything(SEED)

class Pems08_DataModule(pl.LightningDataModule):
    def __init__(self, preimpute_flag="Backward", freq_flag=True):
        super().__init__()
        self.train_loader, self.valid_loader, self.test_loader, self.scaler, self.mean_scaler = get_dataloader(
            batch_size=16, 
            device="cuda:0", 
            val_len=0.1, 
            missing_pattern="point",
            is_interpolate=True, 
            num_workers=16,
            target_strategy="hybrid",
            preimpute_flag=preimpute_flag,
            freq_flag=freq_flag,
        )

    def train_dataloader(self):
        return self.train_loader 
    
    def val_dataloader(self):
        return self.valid_loader
    
    def test_dataloader(self):
        return self.test_loader

if __name__ == "__main__":
    # 1. 实例化 DataModule
    dm = Pems08_DataModule()
    
    # 测试一下秒启动
    print("正在测试数据加载...")
    for batch in dm.train_dataloader():
        print("首个 Batch 加载成功！")
        break


    # 3. 设置 Trainer
    trainer = pl.Trainer(
        max_epochs=200,
        accelerator="gpu",
        devices=1,
        limit_train_batches=50000, 
    )

    # 4. 启动训练
    print("开始训练...")
    # trainer.fit(lightning_model, datamodule=dm)
    # trainer.test(lightning_model, datamodule=dm)