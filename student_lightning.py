import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn, optim
from main_model import OneStep_Model


class StudentLightningModule(pl.LightningModule):
    def __init__(self,config, device, target_dim=36,seq_len=36):
        super(StudentLightningModule, self).__init__()
        self.model = OneStep_Model(config,device,target_dim,seq_len)
        #self.model.load_state_dict(torch.load("/home/duanlei/PriSTI/save/aqi36/model.pth"))
        self.step_counter = 0

    def on_epoch_start(self):
        # Reset the step counter at the start of each epoch
        self.step_counter = 0

#这一datamodule得到的每个batch是一个长度为10(times)的list，其中list[i]是一个字典，每个元素的尺寸为[B,4,36,36] [B,4,K,L]的顺序
    def training_step(self, batch, batch_idx):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
            coeffs,
            cond_mask,
            _
        ) = self.model.process_data(batch)
        side_info = self.model.get_side_info(observed_tp, cond_mask)
        itp_info = coeffs.unsqueeze(1)
        eval_mask = observed_mask - cond_mask

        x_0 = self.model.student_impute(observed_data,cond_mask,side_info,itp_info)
        masked_x0 = x_0*eval_mask
        masked_observed_data = observed_data*eval_mask

        task_loss = F.mse_loss(masked_x0,masked_observed_data)
        self.log('task_loss',task_loss,on_epoch=True,on_step=False)

        if self.step_counter < 100:
            # Print distillation loss for the first five steps of each epoch
            print(f"Epoch {self.current_epoch}, Step {self.step_counter}: Task Loss = {task_loss.item()}")
        self.step_counter += 1

        return task_loss


    def validation_step(self, batch, batch_idx):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
            coeffs,
            cond_mask,
            _
        ) = self.model.process_data(batch)
        side_info = self.model.get_side_info(observed_tp, cond_mask)
        itp_info = coeffs.unsqueeze(1)
        eval_mask = observed_mask - cond_mask
        x_0 = self.model.student_impute(observed_data,cond_mask,side_info,itp_info)
        masked_x0 = eval_mask*x_0


        val_loss = F.mse_loss(masked_x0, observed_data*eval_mask)

        self.log('val_loss', val_loss, on_step=False, on_epoch=True)
        
        return val_loss
    
        # val_loss = F.mse_loss(x_0,observed_data)
        
        # self.log('val_loss',val_loss,on_step=False,on_epoch=True)

        # return val_loss





    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        p1 = int(0.75*100)
        p2 = int(0.9*100)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[p1, p2], gamma=0.1
        )
        return [optimizer], [lr_scheduler]
