import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
from diff_models import Guide_diff
import time
import pickle
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR

class GuideDiffLightning(pl.LightningModule):
    """
    用于训练 Guide_diff 模型的 PyTorch Lightning Module。
    实现了对最终输出和中间层输出的损失计算。
    """
    def __init__(self, model_config, learning_rate=5e-4, final_loss_weight=1.0,epochs=200, decay_flag = False):
        super().__init__()
        # 保存超参数，方便 checkpointing 和 HPO
        self.save_hyperparameters()
        self.epochs = epochs
        # 实例化模型
        self.model = Guide_diff(config=model_config,target_dim=170)
        self.lr = learning_rate
        self.final_loss_weight = final_loss_weight
        self.decay_flag = decay_flag


        # 定义损失函数 (L2 Loss / MSE)
        self.criterion = nn.L1Loss()

        
        self.embed_layer = nn.Embedding(
            num_embeddings=170, embedding_dim=16
        )
        path = "./data/PEMS08/pems08_meanstd.pk"
        with open(path, "rb") as f:
            self.train_mean, self.train_std = pickle.load(f)

    def forward(self, x, diffusion_step, side_info, itp_x, cond_mask):
        # 模型的前向传播，返回最终预测和中间层预测列表
        return self.model(x, diffusion_step, side_info, itp_x, cond_mask)

    def training_step(self, batch, batch_idx):
        # 1. 解包数据
        (
            observed_data,    # [B, K, L]
            observed_mask,    # [B, K, L]
            observed_tp,
            gt_mask,
            cut_length,
            coeffs,
            cond_mask,        # [B, K, L]
            freq_coeffs,
        )=self.process_data(batch)
        side_info = self.get_side_info(observed_tp,cond_mask)

        x_T = torch.randn_like(observed_data)
        total_input = ((1 - cond_mask) * x_T).unsqueeze(1)
        # 2. 模型前向传播
        # final_pred 是最终输出，layer_preds 是中间层输出列表 (假设按深度递增)
        final_pred, layer_preds = self(total_input, torch.tensor([49]).to(self.device), side_info, coeffs.unsqueeze(1), cond_mask)
        
        # 3. 计算最终预测损失 (Final Prediction Loss)
        # final_pred 的形状应该是 (B, K, L)
        
        target_mask = observed_mask - cond_mask
        final_loss = self.criterion(final_pred * target_mask, observed_data * target_mask)
        loss = 0
        layer_loss_list = []
        num_layers = len(layer_preds)

        if num_layers > 0:
            max_weight = 0.1
            gamma = 3.0  # Growth factor for intermediate layer supervision
            
            base_weights = torch.tensor([gamma**(i + 1) for i in range(num_layers)], dtype=torch.float, device=self.device)
            scale_factor = max_weight / base_weights[-1]
            current_weights = base_weights * scale_factor

            weighted_layer_losses = []
            layer_loss_list = []  # To store unweighted losses for debugging

            # Apply decay if decay_flag is True
            if self.decay_flag:
                decay_factor = max(0, 1.0 - self.current_epoch / (self.epochs * 0.5))  # First 50% epochs have strong supervision
            else:
                decay_factor = 1.0  # No decay if decay_flag is False

            for i, pred in enumerate(layer_preds):
                current_weight = current_weights[i]  # Decaying weight for intermediate layers
                
                # Apply weighted error only for intermediate layers
                if i < num_layers - 1:  # All layers except final layer
                    abs_error = torch.abs(observed_data * target_mask - pred * target_mask)
                    weighted_layer_losses.append(decay_factor * current_weight * abs_error.sum()) 
                else:  # Only final layer
                    abs_error = torch.abs(observed_data * target_mask - pred * target_mask)
                    weighted_layer_losses.append(current_weight * abs_error.sum())  # No decay for final layer

            # Sum all weighted losses
            layer_total_weighted_sum = torch.stack(weighted_layer_losses).sum()
            
            if target_mask.sum() > 0:
                layer_weighted_loss = layer_total_weighted_sum / target_mask.sum()
            else:
                layer_weighted_loss = torch.tensor(0.0, device=self.device)

            #loss = self.final_loss_weight * final_loss + layer_weighted_loss
            #loss = final_loss
            loss = layer_weighted_loss
            self.log('layer_weighted_loss', layer_weighted_loss, on_step=False, on_epoch=True)
                
            self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # 1. 解包数据
        (
            observed_data,    # [B, K, L]
            observed_mask,    # [B, K, L]
            observed_tp,
            gt_mask,
            cut_length,
            coeffs,
            cond_mask,        # [B, K, L]
            freq_coeffs,
        )=self.process_data(batch)
        side_info = self.get_side_info(observed_tp,cond_mask)

        x_T = torch.randn_like(observed_data)
        total_input = ((1 - cond_mask) * x_T).unsqueeze(1)
        # 2. 模型前向传播
        final_pred, layer_preds = self(total_input,torch.tensor([49]).to(self.device), side_info, coeffs.unsqueeze(1), cond_mask)
        

        target_mask = observed_mask - cond_mask
        final_loss = self.criterion(final_pred*target_mask, observed_data*target_mask)

   
        total_loss = final_loss

        # 6. 记录日志
        self.log('val_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)

        return total_loss
    
    def on_test_start(self):
        self.cnt = 0
        self.total_mae = 0
        self.total_mse = 0
        self.total_mre = 0
        self.evalpoints_total = 0
        self.total_time = 0
        self.test_start_time = time.time()  # 记录测试开始时间

    def test_step(self,batch,batch_idx):
        
        scaler = torch.from_numpy(self.train_std).to(self.device).float()
        mean_scaler = torch.from_numpy(self.train_mean).to(self.device).float()
        (
            observed_data,    # [B, K, L]
            observed_mask,    # [B, K, L]
            observed_tp,
            gt_mask,
            cut_length,
            coeffs,
            cond_mask,        # [B, K, L]
            freq_coeffs,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask
            side_info = self.get_side_info(observed_tp, cond_mask)
            itp_info = coeffs.unsqueeze(1)
            B,K,L = cond_mask.size()

      
            x_T = torch.randn_like(observed_data)
            diff_input = ((1-cond_mask)*x_T).unsqueeze(1)
            t1 = time.time()
            final_pred, layer_preds = self.model(diff_input,torch.tensor([49]).to(self.device),side_info,itp_info,cond_mask)
            sample = final_pred

            t2 = time.time()
            print("time:",t2-t1)

            
            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0

            final_sample = sample.permute(0,2,1).to(self.device) #(B,L,K)
            c_target = observed_data.permute(0,2,1)
            eval_points = target_mask.permute(0,2,1)
            observed_points = observed_mask.permute(0,2,1)
            
            for i in range(len(cut_length)):  # to avoid double evaluation
                    eval_points[i, ..., 0 : cut_length[i].item()] = 0


            self.test_end_time = time.time()  # 记录测试结束时间
            self.test_duration = self.test_end_time - self.test_start_time  # 计算耗时

         
            mae_current = (
                torch.abs((final_sample-c_target)*eval_points)
            )*scaler
            mse_current = (
                ((final_sample-c_target)*eval_points)**2
            )*(scaler**2)
             # 计算 MRE (Mean Relative Error)
    
            mre_current = (
                torch.abs(((final_sample*scaler) - (c_target*scaler))) / ((torch.abs(c_target)*scaler)+mean_scaler + 1e-8)  # 避免除以零
            ) * eval_points
            

         
           
            self.total_mae += mae_current.sum().item()
            self.total_mse += mse_current.sum().item()
            self.total_mre += mre_current.sum().item()
            self.evalpoints_total += eval_points.sum().item()
          
            self.cnt += 1  # 增加计数器以确保每个batch有唯一的文件名
        
    def on_test_epoch_end(self):
        
        
        print(self.evalpoints_total)
        MAE = self.total_mae / self.evalpoints_total
        MSE = self.total_mse / self.evalpoints_total
        MRE = self.total_mre / self.evalpoints_total
      
        
        # logs
        self.log('test_MAE', MAE)
        self.log('test_MSE', MSE)
        self.log('test_MRE', MRE)
        self.log('test_duration', self.test_duration)  # 记录耗时

        self.MAE = MAE
        self.MSE = MSE
        self.MRE = MRE

        print(f"Test Set Average MAE: {MAE}")
        print(f"Test Set Average MSE: {MSE}")
        print(f"Test Set Average MRE: {MRE}")
        print(f"Test Set Total Duration: {self.test_duration:.2f} seconds")  # 打印总耗时
        pass
    
    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
    


        coeffs = None
        if True:
            coeffs = batch["coeffs"].to(self.device).float()
        cond_mask = batch["cond_mask"].to(self.device).float()
        freq_coeffs = batch["freq"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)  # [B, K, L]
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        cond_mask = cond_mask.permute(0, 2, 1)


        if True:
            coeffs = coeffs.permute(0, 2, 1)
        freq_coeffs = freq_coeffs.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            cut_length,
            coeffs,
            cond_mask,
            freq_coeffs,
        )
    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, 128)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(170).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        return side_info
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Warmup stage: linearly increases over 5 epochs
        warmup_epochs = 5
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: min(epoch / warmup_epochs, 1.0))
        
        # Cosine Annealing
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr * 0.1
        )
        
        # Sequential scheduler: first warmup for `warmup_epochs` epochs, then switch to cosine scheduler
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }