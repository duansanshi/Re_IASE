
from student_lightning import StudentLightningModule
import argparse
import yaml
from dataset_aqi36 import get_dataloader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch 
from main_model import OneStep_Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from REVIN import RevIN

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:1', help='Device for Attack')
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
config["diffusion"]["adj_file"] = 'AQI36'
#config["diffusion"]["adj_file"] = 'pems-bay'
config["seed"] = 42
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)



# train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
#         batch_size=4, device="cuda:1", val_len=0.1,
#         is_interpolate=True, num_workers=16,
#         target_strategy="hybrid", mask_sensor=[]
#     )
# revin_layer = RevIN(num_features=36).to("cuda:1")

# first_train_batch = None
# first_valid_batch = None
# first_test_batch = None
# for batch in train_loader:
#     first_train_batch = batch
#     break
# for batch in valid_loader:
#     first_valid_batch = batch
#     break
# for batch in test_loader:
#     first_test_batch = batch
#     break
# checkpoint_path = '/home/duanlei/PriSTI/student_save/real_student/onestep-aqi-inherited-epoch=05-val_loss=0.00.ckpt'
# student = StudentLightningModule.load_from_checkpoint(checkpoint_path,config=config,device=args.device,target_dim=36,seq_len=36)
# model = student.model

# (
#             observed_data,
#             observed_mask,
#             observed_tp,
#             gt_mask,
#             for_pattern_mask,
#             cut_length,
#             coeffs,
#             cond_mask,
#             student_sample
#         ) = model.process_data(first_train_batch)
# observed_data = revin_layer(observed_data,mode='norm')
# print(observed_data.shape)  #(B,K,L)

# B,K,L = observed_data.shape

# # 将 CUDA 张量转换为 CPU 张量，然后转换为 numpy 数组
# observed_data = observed_data.detach().cpu().numpy()

# print(observed_data.shape)  # 输出 (B, K, L)

# # 先 permute 调整维度顺序，从 (B, K, L) 变成 (B, L, K)
# permuted_data = np.transpose(observed_data, (0, 2, 1))

# # 然后 reshape 为 (B*L, K)
# reshaped_data = permuted_data.reshape(B * L, K)[:, :5]

# # 绘制每个特征并保存为独立的图像文件
# for i in range(5):
#     plt.figure(figsize=(10, 6))
#     plt.plot(reshaped_data[:, i], label=f'Feature {i+1}')
#     plt.xlabel('B*L')
#     plt.ylabel('Feature Values')
#     plt.title(f'Observed Data Reshaped to (B*L, K) for Feature {i+1}')
#     plt.legend()
#     #plt.savefig(f'reshaped_observed_data_feature_{i+1}.png')
#     plt.savefig(f'reshaped_observed_data_feature_valid{i+1}.png')
#     plt.close()


    
# 定义 RevIN 类的参数
num_features = 10  # 假设每个batch有10个特征或通道
B = 5  # 批量大小
L = 20  # 序列长度或其他维度的长度

# 创建一个形状为 (B, K, L) 的随机输入张量
x = torch.randn(B, L, num_features)

# 实例化 RevIN 类
rev_in = RevIN(num_features=num_features, affine=True)

# 将输入张量标准化
x_norm = rev_in(x, mode='norm')

# 打印标准化后的张量形状
print("Normalized tensor shape:", x_norm.shape)

# 假设我们需要将标准化后的张量反标准化
x_denorm = rev_in(x_norm, mode='denorm')

# 打印反标准化后的张量形状
print("Denormalized tensor shape:", x_denorm.shape)