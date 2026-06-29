
from metrla_lightning import MetrlaLightningModule
import argparse
import yaml
from dataset_metrla import get_dataloader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch 
from main_model import OneStep_Model
import numpy as np

parser = argparse.ArgumentParser(description="PriSTI")
parser.add_argument("--config", type=str, default="traffic.yaml")
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
config["diffusion"]["adj_file"] = 'metr-la'
#config["diffusion"]["adj_file"] = 'pems-bay'
config["seed"] = 42
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)



train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
        batch_size=4, device="cuda:1", missing_pattern='block',
        is_interpolate=True, num_workers=12,
        target_strategy='hybrid',
    )




checkpoint_path = '/home/duanlei/PriSTI/student_save/real_student/onestep-metrla-inherited-epoch=09-val_loss=0.00.ckpt'
student = MetrlaLightningModule.load_from_checkpoint(checkpoint_path,config=config,device=args.device,target_dim=207,seq_len=24)

model = student.model
model.eval()

cnt = 0
for batch in train_loader:
    if cnt == 2:
        first = batch
        break
    cnt+=1
(
        observed_data,
        observed_mask,
        observed_tp,
        gt_mask,
        for_pattern_mask,
        cut_length,
        coeffs,
        cond_mask,
    ) = model.process_data_metrla(first)

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

# batch1impute = torch.load("/home/duanlei/PriSTI/batch1studentimpute_metrla.pt")
# itp_info = observed_data*cond_mask+batch1impute*(1-cond_mask)
# itp_info = itp_info.unsqueeze(1)


itp_info = coeffs.unsqueeze(1)
side_info = model.get_side_info(observed_tp,cond_mask)
# #itp_info = (coeffs*cond_mask1).unsqueeze(1)
# #itp_info = coeffs.unsqueeze(1)
# #itp_info = torch.randn(1,1,36,36).to("cuda:0")
# #itp_info = observed_data.unsqueeze(1)
# #itp_info = (observed_data*cond_mask).unsqueeze(1)
    
imputed_samples = model.real_impute(observed_data,cond_mask,side_info,n_samples=10,itp_info=itp_info)
samples_median = imputed_samples.median(dim=1) #(B,K,L)
eval_mask = observed_mask-cond_mask



    
diff1 = torch.abs((samples_median.values-observed_data)*eval_mask)
diff2 = torch.abs((coeffs-observed_data)*eval_mask)
print(diff1.max())
print(diff2.max())
# print(samples_median.values)
# print(observed_data)
print(eval_mask.sum().item())
mae1 = diff1.sum().item()/eval_mask.sum().item()
mae2 = diff2.sum().item()/eval_mask.sum().item()
print(mae1-mae2)

torch.save(samples_median.values.float(), 'batch1studentimpute_metrla.pt')



# mae_total = 0
# eval_points_total = 0
# mse_total = 0
# # 用于存储每次实验的结果
# batch_num = len(test_loader)
# mae_results = [[0 for _ in range(batch_num)] for _ in range(3)]  # 三次实验，每个batch一个值
# mse_results = [[0 for _ in range(batch_num)] for _ in range(3)]  # 同上


# #重复实验三次
# for experiment in range(3):
#     print(f"实验 {experiment + 1}:")

#      # 处理前五个batch
#     for j in range(batch_num):
#         cnt = 0
#         for batch in test_loader:
#             if cnt == j:
#                 first_batch_data = batch
#                 break
#             cnt += 1

#         with torch.no_grad():
            
#             (
#             observed_data,
#             observed_mask,
#             observed_tp,
#             gt_mask,
#             for_pattern_mask,
#             cut_length,
#             coeffs,
#             cond_mask,
#             ) = model.process_data_metrla(batch)

            
#             cond_mask = gt_mask
#             cond_mask = cond_mask.to(observed_data)
            
#             side_info = model.get_side_info(observed_tp,cond_mask)
#             itp_info = coeffs.unsqueeze(1)

#             samples = model.real_impute(observed_data,cond_mask,side_info,n_samples=10,itp_info=itp_info)

            
#             c_target = observed_data
#             eval_points = observed_mask - cond_mask
           

#             samples_median = samples.median(dim=1)
            
#             #scaler形状为[K]，因此先将其它变量尺寸修改为[B,L,K]以方便广播机制运作
#             samples_final = samples_median.values.permute(0,2,1)
#             c_target = c_target.permute(0,2,1)
#             eval_points = eval_points.permute(0,2,1)


#             mae_current = (
#                          torch.abs((samples_final - c_target) * eval_points) 
#                      ) * scaler
#             mse_current = (
#                              ((samples_final - c_target) * eval_points) ** 2
#                          ) * (scaler ** 2)
#             mae_total += mae_current.sum().item()
#             eval_points_total += eval_points.sum().item()
#             mse_total += mse_current.sum().item()
#             MAE = torch.mean(mae_current[mae_current != 0].float())
#             MSE = torch.mean(mse_current[mse_current != 0].float())

#             # 存储结果
#             mae_results[experiment][j] = MAE.item()
#             mse_results[experiment][j] = MSE.item()

#  # 计算每个batch三次实验的平均MAE和MSE
# for j in range(batch_num):
#     avg_mae = sum(mae_results[experiment][j] for experiment in range(3)) / 3
#     avg_mse = sum(mse_results[experiment][j] for experiment in range(3)) / 3
#     print(f"batch {j+1} 的 MAE 平均值: {avg_mae:.4f}")
#     print(f"batch {j+1} 的 MSE 平均值: {avg_mse:.4f}")

# mae_total/=3
# mse_total/=3
# eval_points_total/=3
# print(eval_points_total)
# print(mae_total/eval_points_total)
# print(mse_total/eval_points_total)