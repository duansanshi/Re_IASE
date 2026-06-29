import pytorch_lightning as pl
import torch
import torch.nn as nn
import pickle
from torch.optim.lr_scheduler import LambdaLR, ChainedScheduler


class aqi_lightning_module(pl.LightningModule):
    def __init__(self, model, lr=1e-3, seq_len=36, feature_dim=36, loss_flag="l1",epochs=200):
        super().__init__()
        path = "/home/duanlei/PriSTI/data/pm25/pm25_meanstd.pk"
        with open(path, "rb") as f:
            self.train_mean, self.train_std = pickle.load(f)
        self.model = model
        self.lr = lr
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.loss_flag = loss_flag  # "l1" or "threshold" or "midSupervision"
        self.epochs = epochs

    def training_step(self, batch, batch_idx):
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Extract data from batch using the model's process_data method
        (
            observed_data,    # [B, K, L]
            observed_mask,    # [B, K, L]
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            coeffs,
            cond_mask,        # [B, K, L]
            freq_coeffs,
            timeofday,
            dayofweek
        ) = self.model.process_data(batch)
        
        # Prepare input: mask observed data with conditional mask
        x = observed_data * cond_mask  # [B, K, L]
        
        # Forward pass through the model
        #imputed_results = self.model(x,cond_mask)  # [B, K, L]
        #imputed_results = self.model(coeffs,cond_mask)  # [B, K, L]
        #imputed_results,layer_preds = self.model(freq_coeffs,cond_mask,timeofday,dayofweek)  # [B, K, L]
        noise = torch.randn_like(observed_data) # [B, K, L]
        imputed_results = self.model(noise,freq_coeffs,cond_mask, torch.tensor([99]*16).to(self.device).unsqueeze(1))   # [B, K, L]
    
        # Compute evaluation mask (positions to evaluate imputation)
        eval_mask = observed_mask - cond_mask  # [B, K, L]

        # Calculate MSE loss only on imputed positions
        #loss = ((observed_data - imputed_results) ** 2 * eval_mask).sum() / eval_mask.sum() if eval_mask.sum()>0 else 1
        if self.loss_flag == "l1":
            loss = (torch.abs(observed_data - imputed_results) * eval_mask).sum() / eval_mask.sum() if eval_mask.sum()>0 else 1

        elif self.loss_flag == "threshold":       
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
    
        elif self.loss_flag == "midSupervision":
            loss = 0
            #loss = (torch.abs(observed_data - imputed_results) * eval_mask).sum() / eval_mask.sum() if eval_mask.sum()>0 else 1
            #loss = ((observed_data - imputed_results) ** 2 * eval_mask).sum() / eval_mask.sum() if eval_mask.sum()>0 else 1
            layer_loss_list = []
            num_layers = len(layer_preds)
            
        

            if num_layers > 0:
                max_weight = 0.1
                # print(max_weight) # 调试信息可以删除
                gamma = 3.0 # 增长因子，可以调整 (e.g., 1.5, 2.0, 3.0)

                if num_layers > 0:
                    # 1. 计算和归一化 current_weights 的逻辑保持不变，确保 current_weights[-1] == max_weight
                    base_weights = torch.tensor([gamma**(i + 1) for i in range(num_layers)], 
                                                dtype=torch.float, device=self.device)
                    scale_factor = max_weight / base_weights[-1]
                    current_weights = base_weights * scale_factor
                    
                    weighted_layer_losses = []
                    # layer_loss_list 暂时保留，用于记录 unweighted loss，方便打印和调试
                    layer_loss_list = [] 
                    
                    for i, pred in enumerate(layer_preds):
                        current_weight = current_weights[i]
                        
                        # ⚠️ 注意：layer_loss 仍然是 MAE，但我们现在需要它的总和（非平均）
                        # 计算每个时间步和特征的绝对误差
                        abs_error = torch.abs(observed_data* eval_mask - pred* eval_mask)
                        
                        # 这一步计算的是所有评估点的加权绝对误差的总和
                        weighted_layer_losses.append(current_weight * abs_error.sum()) 
                        
                        # 记录未加权的平均 MAE，用于调试和日志
                        layer_loss_unweighted = abs_error.sum() / eval_mask.sum() if eval_mask.sum() > 0 else torch.tensor(0.0, device=self.device)
                        layer_loss_list.append(layer_loss_unweighted)


                    # 1. 计算总加权损失 (Total Weighted Loss)
                    # 将所有层的加权总误差求和
                    # sum() / eval_mask.sum() 这一步被分解到了权重和 sum() 中，避免重复除以 eval_mask.sum()
                    layer_total_weighted_sum = torch.stack(weighted_layer_losses).sum()

                    # 2. 将总加权误差归一化（除以总评估点数）
                    # 这样 layer_weighted_loss 的量纲和主 loss (MAE) 保持一致
                    if eval_mask.sum() > 0:
                        layer_weighted_loss = layer_total_weighted_sum / eval_mask.sum()
                    else:
                        layer_weighted_loss = torch.tensor(0.0, device=self.device)

                    # 3. 最终合并：使用加权损失
                    layer_mean_unweighted_loss = torch.stack(layer_loss_list).mean() 

                    # print(layer_weighted_loss) # 调试
                    # print(layer_mean_unweighted_loss) # 调试
                    
                    # 最终总 Loss = 主 Loss + 加权 Layer Loss
                    loss += layer_weighted_loss 
                    
                    #建议 log 记录加权和非加权的 layer loss
            self.log('layer_weighted_loss', layer_weighted_loss, on_step=False, on_epoch=True)
            self.log('layer_unweighted_mean_loss', layer_mean_unweighted_loss, on_step=False, on_epoch=True)
        

        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Same data extraction as in training_step
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
        ) = self.model.process_data(batch)

        # Prepare input
        x = observed_data * cond_mask  # [B, K, L]
        
     
        noise = torch.randn_like(observed_data) # [B, K, L]
        imputed_results= self.model(noise,freq_coeffs,cond_mask, torch.tensor([99]*16).to(self.device).unsqueeze(1))  # [B, K, L]
        # Compute evaluation mask
        eval_mask = observed_mask - cond_mask  # [B, K, L]

        # Calculate validation loss
        loss = ((observed_data - imputed_results) ** 2 * eval_mask).sum() / eval_mask.sum()
        
        # Log validation loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_test_start(self):
        self.total_mae = 0
        self.total_mse = 0
        self.evalpoints_total = 0
    
    def test_step(self, batch, batch_idx):
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
        ) = self.model.process_data(batch)

        x = observed_data * cond_mask
        noise = torch.randn_like(observed_data) # [B, K, L]
        imputed_results= self.model(noise,freq_coeffs,cond_mask, torch.tensor([99]*16).to(self.device).unsqueeze(1))  # [B, K, L]

        #############
        imputed_results = imputed_results.permute(0,2,1)  # [B, L, K] 
        eval_mask = observed_mask - cond_mask
        # for i in range(len(cut_length)):  # to avoid double evaluation
        #             eval_mask[i, ..., 0 : cut_length[i].item()] = 0
        eval_mask = eval_mask.permute(0,2,1)  # [B, L, K]
        observed_data = observed_data.permute(0,2,1)  # [B, L, K]
        #############


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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # 预热阶段：5 个 epoch 线性增加
        warmup_epochs = 5
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: min(epoch / warmup_epochs, 1.0))
        
        # 余弦退火
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr * 0.1
        )
        
        # 组合调度器
        scheduler = ChainedScheduler([warmup_scheduler, cosine_scheduler])
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }