from pemsbay_datamodule import PemsBayDataModule
from pemsbay_lightning import PemsBayLightningModule
import argparse
import yaml
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping   
import torch.nn.functional as F
import torch


parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="traffic.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument(
    "--targetstrategy", type=str, default="hybrid", choices=["hybrid", "random", "historical"]
)
parser.add_argument("--nsample", type=int, default=100)


args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = False
config["model"]["target_strategy"] = args.targetstrategy
config["diffusion"]["adj_file"] = 'pems-bay'
config["seed"] = 42

seed_everything(24)

student = PemsBayLightningModule(config,args.device,target_dim=325,seq_len=24)
dm = PemsBayDataModule()



early_stop_callback = EarlyStopping(
   monitor='val_loss',  # 监控验证集的损失
   min_delta=0.00,  # 指标改进的最小变化
   patience=20,  # 10个 epochs 内无改进则停止
   verbose=True,  # 打印停止的消息
   mode='min'  # 监控指标应该是最小化
)

# 设置模型检查点
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',     
    dirpath='/home/duanlei/PriSTI/student_save/real_student',    # 检查点文件保存的目录
    filename='onestep-pemsbay-inherited-{epoch:02d}-{val_loss:.2f}',  # 文件命名规则
    save_top_k=1,          
    mode='min' ,             # 'min' 模式表示损失越小越好
    save_last = True
)


trainer = pl.Trainer(
    max_epochs = 100,
    gpus = [0],
    progress_bar_refresh_rate = 1,
    check_val_every_n_epoch=1,
    callbacks=[checkpoint_callback,early_stop_callback]
)

trainer.fit(student,dm)