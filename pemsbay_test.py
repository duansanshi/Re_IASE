
from pemsbay_lightning import PemsBayLightningModule
import argparse
import yaml
from dataset_pemsbay import get_dataloader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch 
from main_model import OneStep_Model


parser = argparse.ArgumentParser(description="PriSTI")
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




train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
        batch_size=4, device="cuda:0", missing_pattern='block',
        is_interpolate=True, num_workers=4,
        target_strategy='hybrid',
    )




checkpoint_path = '/home/duanlei/PriSTI/student_save/real_student/onestep-pemsbay-random-epoch=40-val_loss=0.01.ckpt'
student = PemsBayLightningModule.load_from_checkpoint(checkpoint_path,config=config,device=args.device,target_dim=325,seq_len=24)

model = student.model
model.eval()





mae_total = 0
eval_points_total = 0
# 用于存储每次实验的结果
batch_num = len(test_loader)
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
            for_pattern_mask,
            cut_length,
            coeffs,
            cond_mask,
            ) = model.process_data_pemsbay(batch)

            
            cond_mask = gt_mask
            cond_mask = cond_mask.to(observed_data)
            
            side_info = model.get_side_info(observed_tp,cond_mask)
            itp_info = coeffs.unsqueeze(1)

            samples = model.real_impute(observed_data,cond_mask,side_info,n_samples=10,itp_info=itp_info)

            
            c_target = observed_data
            eval_points = observed_mask - cond_mask
           

            samples_median = samples.median(dim=1)
            
            #scaler形状为[K]，因此先将其它变量尺寸修改为[B,L,K]以方便广播机制运作
            samples_final = samples_median.values.permute(0,2,1)
            c_target = c_target.permute(0,2,1)
            eval_points = eval_points.permute(0,2,1)


            mae_current = (
                         torch.abs((samples_final - c_target) * eval_points) 
                     ) * scaler
            mse_current = (
                             ((samples_final - c_target) * eval_points) ** 2
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
    print(f"batch {j+1} 的 MSE 平均值: {avg_mse:.4f}")

mae_total/=3
eval_points_total/=3
print(eval_points_total)
print(mae_total/eval_points_total)