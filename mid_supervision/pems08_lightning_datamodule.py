import sys
sys.path.append("./")

from dataset_pems08 import get_dataloader
import pytorch_lightning as pl
import numpy as np
import torch
import yaml


SEED = 42
pl.seed_everything(SEED) # 推荐使用 Lightning 的全局随机种子设置

train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
    batch_size=16, 
    device="cuda:0", 
    val_len=0.1, 
    missing_pattern="point",
    is_interpolate=True, 
    num_workers=16, # 建议设为 4 或 8
    target_strategy="hybrid",
)

class Pems08_DataModule(pl.LightningDataModule):
    def train_dataloader(self):
        return train_loader 
    
    def val_dataloader(self):
        return valid_loader
    
    def test_dataloader(self):
        return test_loader

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