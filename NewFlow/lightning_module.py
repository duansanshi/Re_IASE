import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import pickle


class AQI_LightningModule(pl.LightningModule):
    def __init__(self, model,id, lr=1e-3, criterion=None):
        super().__init__()
        path = "/home/duanlei/PriSTI/data/pm25/pm25_meanstd.pk"
        with open(path, "rb") as f:
            self.train_mean, self.train_std = pickle.load(f)
        self.model = model
        self.lr = lr
        self.flag = "pristi"
        self.steps = 3
        self.n_samples = 20
        self.MAE = 0

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
        )=self.model.process_data(batch)
        
        if self.flag == "pristi":
            side_info = self.model.get_side_info(observed_tp,cond_mask)
            t = torch.rand(observed_data.shape[0], 1, 1, device=self.model.device)
            t_partial = (t*99).to(self.model.device).squeeze()
            # t = torch.rand(1).to(self.model.device)
            # t_partial = (t*99).to(self.model.device)
    
            X_0 = torch.randn_like(observed_data)
            #X_0 = freq_coeffs
            X_t = (1-t)*X_0 + t*observed_data
            total_input = (X_t).unsqueeze(1)
            
            velocity = self.model(total_input,side_info,t_partial,freq_coeffs.unsqueeze(1),cond_mask)
        else:
            t = torch.rand(observed_data.shape[0], 1, 1, device=self.model.device)
            t_partial = (t*99).to(self.model.device).squeeze()

    
            X_0 = torch.randn_like(observed_data)
            #X_0 = freq_coeffs
            X_t = (1-t)*X_0 + t*observed_data
            total_input = (X_t)
            
            velocity = self.model(freq_coeffs,cond_mask,total_input,t_partial)
        

        eval_mask = observed_mask-cond_mask
        num_eval = eval_mask.sum()
        
        loss = (torch.abs(velocity-(observed_data-X_0))**2*eval_mask).sum()/(num_eval if num_eval > 0 else 1)
        #loss = (torch.abs(velocity-(observed_data-X_0))*eval_mask).sum()/(num_eval if num_eval > 0 else 1)
        
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
         
        )=self.model.process_data(batch)
  
        if self.flag == "pristi":
            side_info = self.model.get_side_info(observed_tp,cond_mask)
            t = torch.rand(1).to(self.model.device)
            t_partial = (t*99).to(self.model.device)
            X_0 = torch.randn_like(observed_data)
            #X_0 = freq_coeffs
            X_t = (1-t)*X_0 + t*observed_data
            total_input = (X_t).unsqueeze(1)
            velocity = self.model(total_input,side_info,torch.tensor([t_partial]),freq_coeffs.unsqueeze(1),cond_mask)
        else:
            t = torch.rand(1).to(self.model.device)
            t_partial = (t*99).to(self.model.device)
            X_0 = torch.randn_like(observed_data)
            #X_0 = freq_coeffs
            X_t = (1-t)*X_0 + t*observed_data
            total_input = (X_t)
            velocity = self.model(freq_coeffs,cond_mask,total_input,t_partial)

        eval_mask = observed_mask-cond_mask
        num_eval = eval_mask.sum()
        #loss = ((velocity-(observed_data-X_0))**2*eval_mask).sum()/(num_eval if num_eval > 0 else 1)
        loss = (torch.abs(velocity-(observed_data-X_0))*eval_mask).sum()/(num_eval if num_eval > 0 else 1)
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
        )=self.model.process_data(batch)

        with torch.no_grad():
            B, K, L = observed_data.shape
            n_samples = self.n_samples
            imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

            for j in range(n_samples):
            
                cond_mask = gt_mask
                eval_mask = observed_mask - cond_mask
                if self.flag == 'pristi':
                    side_info = self.model.get_side_info(observed_tp,cond_mask)
                # 初始状态：t=0 对应 freq_coeffs
                X_t = torch.randn_like(observed_data)
                #X_t = freq_coeffs
                steps = self.steps  # ODE 积分步数
                ts = torch.linspace(0.0, 1.0, steps+1).to(self.device)

                for i in range(steps):
                    t_cur = ts[i]
                    t_next = ts[i+1]
                    dt = t_next - t_cur

                    if self.flag == 'pristi':
                        velocity = self.model(
                            X_t.unsqueeze(1),
                            side_info,
                            torch.tensor([t_cur*99], device=self.device),
                            freq_coeffs.unsqueeze(1),
                            cond_mask,
                        )
                    else:
                        velocity = self.model(freq_coeffs,cond_mask,X_t,torch.tensor([t_cur*99], device=self.device))

                    # Euler 更新
                    X_t = X_t + dt * velocity.squeeze(1)

                imputed_samples[:,j] = X_t
                    


            for i in range(len(cut_length)):  # to avoid double evaluation
                eval_mask[i, ..., 0 : cut_length[i].item()] = 0

            imputed_results = torch.median(imputed_samples,dim=1).values
            imputed_results = imputed_results.permute(0,2,1)  # [B, L, K]
            observed_data = observed_data.permute(0,2,1)  # [B, L, K]
            eval_mask = eval_mask.permute(0,2,1)  # [B, L, K]
            
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
        self.MAE = MAE

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),lr=self.lr)
        return optimizer
    
   
