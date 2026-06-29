
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



train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
        batch_size=4, device="cuda:1", val_len=0.1,
        is_interpolate=True, num_workers=16,
        target_strategy="hybrid", mask_sensor=[]
    )




checkpoint_path = '/home/duanlei/PriSTI/student_save/real_student/onestep-aqi-inherited-epoch=05-val_loss=0.00.ckpt'
student = StudentLightningModule.load_from_checkpoint(checkpoint_path,config=config,device=args.device,target_dim=36,seq_len=36)

model = student.model
# torch.save(model.state_dict(),"student.pth")
model.eval()

# # 定义一个函数来处理数据并保存为 DataF rame
# def process_and_save_data(loader, filename):
#     # 初始化一个空的 DataFrame
#     data_df = pd.DataFrame()

#     for batch in loader:
#         (
#             observed_data,
#             observed_mask,
#             observed_tp,
#             gt_mask,
#             for_pattern_mask,
#             cut_length,
#             coeffs,
#             cond_mask,
#             student_sample
#         ) = model.process_data(batch)
        
#         print(observed_data)
#         print(student_sample)
#         print(cond_mask)
#         return 0
#         side_info = model.get_side_info(observed_tp, cond_mask)
#         itp_info = coeffs.unsqueeze(1)
#         imputed_samples = model.real_impute(observed_data, cond_mask, side_info, n_samples=100, itp_info=itp_info)

#         # 获取 median 并调整维度
#         samples_median = imputed_samples.median(dim=1).values  # (B, K, L)
#         samples_median = samples_median.permute(0, 2, 1)  # (B, L, K)

#         # # 将 (B, L, K) 转换为 (B * L, K)
#         # samples_median_np = samples_median.reshape(-1, samples_median.shape[-1]).cpu().numpy()

#         # # 将 numpy 数组转换为 DataFrame
#         # samples_median_df = pd.DataFrame(samples_median_np)

#         # # 将批次的 DataFrame 追加到主 DataFrame
#         # data_df = pd.concat([data_df, samples_median_df], ignore_index=True)

#     # 保存 DataFrame 到 CSV 文件
#     #data_df.to_csv(filename, index=False)

# # 处理训练数据
# # process_and_save_data(train_loader, 'train_samples_median.csv')

# # # 处理验证数据
# # process_and_save_data(valid_loader, 'valid_samples_median.csv')

# # # 处理测试数据
# # process_and_save_data(test_loader, 'test_samples_median.csv')

# for batch in valid_loader:
#     (
#         observed_data,
#         observed_mask,
#         observed_tp,
#         gt_mask,
#         for_pattern_mask,
#         cut_length,
#         coeffs,
#         cond_mask,
#         student_sample
#     ) = model.process_data(batch)

#泄露部分信息
# mask = torch.ones_like(cond_mask)
# index = torch.rand_like(cond_mask)
# mask[index<0.5]=0
# cond_mask1 = cond_mask + mask
# cond_mask1[cond_mask1==2]=1

# #cond_mask=0处补值
# x = torch.ones_like(cond_mask)
# itp_info = observed_data*cond_mask + 0.7*x*(1-cond_mask)
# print(itp_info)
# itp_info = itp_info.unsqueeze(1)

# batch1impute = torch.load("/home/duanlei/myPriSTI/batch1impute.pt")
# itp_info = observed_data*cond_mask+batch1impute*(1-cond_mask)
# itp_info = itp_info.unsqueeze(1)


#     itp_info = coeffs.unsqueeze(1)
#     side_info = model.get_side_info(observed_tp,cond_mask)
# # #itp_info = (coeffs*cond_mask1).unsqueeze(1)
# # #itp_info = coeffs.unsqueeze(1)
# # #itp_info = torch.randn(1,1,36,36).to("cuda:0")
# # #itp_info = observed_data.unsqueeze(1)
# # #itp_info = (observed_data*cond_mask).unsqueeze(1)
    
#     imputed_samples = model.real_impute(observed_data,cond_mask,side_info,n_samples=10,itp_info=itp_info)
#     samples_median = imputed_samples.median(dim=1) #(B,K,L)
#     eval_mask = observed_mask-cond_mask



    
#     diff1 = torch.abs((samples_median.values-observed_data)*eval_mask)
#     diff2 = torch.abs((coeffs-observed_data)*eval_mask)
#     print(diff1.max())
#     print(diff2.max())
#     # print(samples_median.values)
#     # print(observed_data)
#     print(eval_mask.sum().item())
#     mae1 = diff1.sum().item()/eval_mask.sum().item()
#     mae2 = diff2.sum().item()/eval_mask.sum().item()
#     print(mae1-mae2)

# torch.save(samples_median.values.float(), 'batch1studentimpute.pt')





mae_total = 0
eval_points_total = 0
# 用于存储每次实验的结果
batch_num = 21
mae_results = [[0 for _ in range(batch_num)] for _ in range(3)]  # 三次实验，每个batch一个值
mse_results = [[0 for _ in range(batch_num)] for _ in range(3)]  # 同上


#重复实验三次
for experiment in range(3):
    print(f"实验 {experiment + 1}:")

     # 处理前五个batch
    for j in range(batch_num):
        cnt = 0
        for batch in test_loader:
            if cnt == j:
                first_batch_data = batch
                break
            cnt += 1

        with torch.no_grad():
            
            (
             observed_data,
             observed_mask,
             observed_tp,
             gt_mask,
             _,
             cut_length,
             coeffs,
             _,
             _
             ) = model.process_data(batch)

            
            cond_mask = gt_mask
            cond_mask = cond_mask.to(observed_data)
            
            side_info = model.get_side_info(observed_tp,cond_mask)
            itp_info = coeffs.unsqueeze(1)

            samples = model.real_impute(observed_data,cond_mask,side_info,n_samples=10,itp_info=itp_info)

            
            c_target = observed_data
            eval_points = observed_mask - cond_mask

            for i in range(len(cut_length)):  # to avoid double evaluation
                eval_points[i, ..., 0 : cut_length[i].item()] = 0

            samples_median = samples.median(dim=1)
           
            mae_current = (
                         torch.abs((samples_median.values - c_target) * eval_points) 
                     ) * scaler
            mse_current = (
                             ((samples_median.values - c_target) * eval_points) ** 2
                         ) * (scaler ** 2)
            mae_total += mae_current.sum().item()
            eval_points_total += eval_points.sum().item()
            MAE = torch.mean(mae_current[mae_current != 0].float())
            MSE = torch.mean(mse_current[mse_current != 0].float())

            # 存储结果
            mae_results[experiment][j] = MAE.item()
            mse_results[experiment][j] = MSE.item()

 # 计算每个batch三次实验的平均MAE和MSE
for j in range(batch_num):
    avg_mae = sum(mae_results[experiment][j] for experiment in range(3)) / 3
    avg_mse = sum(mse_results[experiment][j] for experiment in range(3)) / 3
    print(f"batch {j+1} 的 MAE 平均值: {avg_mae:.4f}")
    #print(f"batch {j+1} 的 MSE 平均值: {avg_mse:.4f}")

mae_total/=3
eval_points_total/=3
print(eval_points_total)
print(mae_total/eval_points_total)