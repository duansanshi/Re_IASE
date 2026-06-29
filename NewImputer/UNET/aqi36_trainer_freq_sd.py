from aqi36_lightning_datamodule import AQI36_DataModule
from imputer_lightning_module_freq_sd import aqi_lightning_module
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import random
import torch
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 2026
set_seed(SEED)

#############################
# 实验配置
#############################
train_flag = False
test_flag = True          # 训练后是否进行测试
device_num = 0
epochs = 200
device = f"cuda:{device_num}"

# 仅测试模式（train_flag=False）时，指定检查点路径
checkpoint_path = "/home/duanlei/PriSTI/checkpoints/myimputer_aqi36_freq_aux/last.ckpt"    # 例如: "checkpoints/myimputer_aqi36_stochastic_depth/best_model.ckpt"

# 选择实验模式：
#   "freq"      -> 原模型 + 频域辅助损失
#   "sd"        -> 随机深度模型 + L1损失
#   "freq_sd"   -> 随机深度模型 + 频域辅助损失
experiment = "sd"

#############################
# 构建模型和 lightning module
#############################
aqi36_dm = AQI36_DataModule(device=device)

if experiment == "freq":
    # 方案3：原模型 + 频域损失
    from transformer_imputer_new import transformer_imputer
    model = transformer_imputer(device=device)
    loss_flag = "freqAux"
    exp_name = "freq_aux"
    
elif experiment == "sd":
    # 方案4：随机深度模型 + L1
    from transformer_imputer_sd import transformer_imputer_sd
    model = transformer_imputer_sd(device=device, drop_path_rate=0.2)
    loss_flag = "l1"
    exp_name = "stochastic_depth"
    
elif experiment == "freq_sd":
    # 方案3+4：随机深度 + 频域损失
    from transformer_imputer_sd import transformer_imputer_sd
    model = transformer_imputer_sd(device=device, drop_path_rate=0.2)
    loss_flag = "freqAux"
    exp_name = "freq_sd"

if train_flag:
    aqi36_imputer = aqi_lightning_module(
        model, lr=5e-4, loss_flag=loss_flag, epochs=epochs,
        decay_flag=False, freq_loss_weight=0.1
    )
else:
    # 仅测试模式：必须指定 checkpoint_path
    if checkpoint_path is None:
        raise ValueError("当 train_flag=False 时，必须指定 checkpoint_path 参数")
    print(f"Loading model from checkpoint: {checkpoint_path}")
    aqi36_imputer = aqi_lightning_module.load_from_checkpoint(
        checkpoint_path=checkpoint_path, model=model, lr=5e-4
    )

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=f"checkpoints/myimputer_aqi36_{exp_name}",
    filename="best_model",
    save_top_k=1,
    save_last=True,
    mode="min",
    verbose=True
)

logger = TensorBoardLogger(save_dir="logs", name=f"unet_imputer_aqi36_{exp_name}")

trainer = Trainer(
    devices=[device_num],
    max_epochs=epochs,
    callbacks=[checkpoint_callback],
    deterministic=True,
    logger=logger,
)

if train_flag:
    trainer.fit(aqi36_imputer, aqi36_dm)
    # 训练完成后，使用最优模型进行测试
    if test_flag:
        best_model_path = checkpoint_callback.best_model_path
        print(f"\nLoading best model from {best_model_path}")
        aqi36_imputer = aqi_lightning_module.load_from_checkpoint(
            checkpoint_path=best_model_path, model=model, lr=5e-4
        )
        # 运行测试
        print("\n" + "="*50)
        print("Running Test...")
        print("="*50)
        trainer.test(aqi36_imputer, aqi36_dm)
    else:
        print("\nTraining completed. Test skipped (test_flag=False)")
else:
    # 仅测试模式
    if test_flag:
        print("\n" + "="*50)
        print("Running Test (test_flag=True, train_flag=False)...")
        print("="*50)
        trainer.test(aqi36_imputer, aqi36_dm)
    else:
        print("Both train_flag and test_flag are False. Nothing to do.")
