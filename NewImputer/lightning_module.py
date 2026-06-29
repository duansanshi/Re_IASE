import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

flag_list = ["pristi","csdi"]
flag = flag_list[1]
class AQI_LightningModule(pl.LightningModule):
    def __init__(self, model,id, lr=1e-3, criterion=None):
        super().__init__()
        path = "/home/duanlei/PriSTI/data/pm25/pm25_meanstd.pk"
        with open(path, "rb") as f:
            self.train_mean, self.train_std = pickle.load(f)
        self.model = model
        self.lr = lr
        self.flag = flag

    def training_step(self, batch, batch_idx):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            coeffs,
            cond_mask,
            freq_coeffs,
            timeofday,
            dayofweek
        )=self.model.process_data(batch)
        
        if self.flag == "pristi":
            side_info = self.model.get_side_info(observed_tp,cond_mask,timeofday,dayofweek)
            noisy_data = torch.randn_like(observed_data)
            total_input = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            imputed_results = self.model(total_input,side_info,torch.tensor([99]),coeffs.unsqueeze(1),cond_mask)
        else:
            side_info = self.model.get_side_info(observed_tp,cond_mask,timeofday,dayofweek)
            noisy_data = torch.randn_like(observed_data)
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            # total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
            total_input = torch.cat([cond_obs, noisy_target,cond_mask.clone().unsqueeze(1)], dim=1)  # (B,3,K,L)
            imputed_results = self.model(total_input, side_info,49)

        eval_mask = observed_mask-cond_mask
        num_eval = eval_mask.sum()
        #loss = ((observed_data-imputed_results)**2*eval_mask).sum()/(num_eval if num_eval > 0 else 1)
        #loss = (torch.abs(observed_data-imputed_results)*eval_mask).sum()/(num_eval if num_eval > 0 else 1)
        # # 定义阈值和放大因子
        theta = 0.5  # 示例阈值，可根据数据调整
        alpha = 0.5  # 示例放大因子，对超过阈值的误差放大两倍

        # 计算绝对误差
        #abs_errors = torch.abs(observed_data - imputed_results)**2 * eval_mask
        abs_errors = torch.abs(observed_data - imputed_results)* eval_mask

        # 创建超过阈值的掩码
        threshold_mask = (abs_errors > theta).float()

        # 调整误差：对超过阈值的部分应用放大因子
        adjusted_errors = abs_errors * (1 + (alpha - 1) * threshold_mask)

        # 计算损失
        loss = adjusted_errors.sum() / eval_mask.sum() if eval_mask.sum() > 0 else torch.tensor(1.0, device=observed_data.device)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            coeffs,
            cond_mask,
            freq_coeffs,
            timeofday,
            dayofweek
        )=self.model.process_data(batch)
  
        if self.flag == "pristi":
            side_info = self.model.get_side_info(observed_tp,cond_mask)
            noisy_data = torch.randn_like(observed_data)
            total_input = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            imputed_results = self.model(total_input,side_info,torch.tensor([99]),coeffs.unsqueeze(1),cond_mask)
        else:
            side_info = self.model.get_side_info(observed_tp,cond_mask,timeofday,dayofweek)
            noisy_data = torch.randn_like(observed_data)
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            #total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
            total_input = torch.cat([cond_obs, noisy_target,cond_mask.clone().unsqueeze(1)], dim=1)  # (B,3,K,L)
            imputed_results = self.model(total_input, side_info,49)

        eval_mask = observed_mask-cond_mask
        num_eval = eval_mask.sum()
        loss = ((observed_data-imputed_results)**2*eval_mask).sum()/(num_eval if num_eval > 0 else 1)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss
    
    def on_test_start(self):
        self.total_mae = 0
        self.total_mse = 0
        self.evalpoints_total = 0

    def test_step(self,batch,batch_idx):
        
        scaler = torch.from_numpy(self.train_std).to(self.device).float()
        mean_scaler = torch.from_numpy(self.train_mean).to(self.device).float()
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            coeffs,
            cond_mask,
            freq_coeffs,
            timeofday,
            dayofweek
        )=self.model.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            eval_mask = observed_mask - cond_mask
            prob = 0.5
            random_values = torch.rand_like(cond_mask)
            cond_mask_half = cond_mask.clone()
            cond_mask_half[random_values<prob] = 1 
       
            
          
           
            B,K,L = observed_data.shape
            imputed_samples = torch.zeros(B, 10, K, L).to(self.device)
            if self.flag  == "pristi":
                side_info = self.model.get_side_info(observed_tp,cond_mask)
                B,K,L = observed_data.shape
                imputed_samples = torch.zeros(B, 10, K, L).to(self.device)
                for i in range(10):
                    noisy_data = torch.randn_like(observed_data)
                    total_input = ((1 - cond_mask) * noisy_data).unsqueeze(1)
                    imputed_samples[:,i] = self.model(total_input,side_info,torch.tensor([99]),coeffs.unsqueeze(1),cond_mask)
                    half_result =  self.model(total_input,side_info,torch.tensor([99]),coeffs.unsqueeze(1),cond_mask)*cond_mask_half+(1-cond_mask_half)*coeffs
                    noisy_data_half = torch.randn_like(observed_data)
                    total_input_half = ((1 - cond_mask_half) * noisy_data_half).unsqueeze(1)
                    imputed_samples[:,i] = self.model(total_input_half,side_info,torch.tensor([99]),half_result.unsqueeze(1),cond_mask_half)
                imputed_results = torch.median(imputed_samples,dim=1).values
            elif self.flag  == "csdi":      
                side_info = self.model.get_side_info(observed_tp,cond_mask,timeofday,dayofweek)
                B,K,L = observed_data.shape
                imputed_samples = torch.zeros(B, 10, K, L).to(self.device)
                for i in range(10):
                    noisy_data = torch.randn_like(observed_data)
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
                    #total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                    total_input = torch.cat([cond_obs, noisy_target,cond_mask.clone().unsqueeze(1)], dim=1)  # (B,3,K,L)
                    imputed_samples[:,i] = self.model(total_input, side_info,49)
            imputed_results = torch.median(imputed_samples,dim=1).values

            for i in range(len(cut_length)):  # to avoid double evaluation
                eval_mask[i, ..., 0 : cut_length[i].item()] = 0


            mae_current = (
                torch.abs((imputed_results-observed_data)*eval_mask)
            )*scaler
            mse_current = (
                ((imputed_results-observed_data)*eval_mask)**2
            )*(scaler**2)
            
            self.total_mae += mae_current.sum().item()
            self.total_mse += mse_current.sum().item()
            self.evalpoints_total += eval_mask.sum().item()
    
    def on_test_epoch_end(self):
        print(self.evalpoints_total)
        MAE = self.total_mae / self.evalpoints_total
        MSE = self.total_mse / self.evalpoints_total
        
        # logs
        self.log('test_MAE', MAE)
        self.log('test_MSE', MSE)

        print(f"Test Set Average MAE: {MAE}")
        print(f"Test Set Average MSE: {MSE}")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),lr=self.lr)
        return optimizer
    
   
