import numpy as np
import torch
import torch.nn as nn
from diff_model import transformer_imputer


class PriSTI(nn.Module):
    def __init__(self, target_dim, seq_len, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim
        self.seq_len = seq_len

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]
        self.use_guide = config["model"]["use_guide"]

        self.cde_output_channels = config["diffusion"]["channels"]
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        ).to(device)

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim
        config_diff["device"] = device
        self.device = device

        input_dim = 2
        self.diffmodel = transformer_imputer(device).to(device)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe


    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, coeffs, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, coeffs, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, coeffs,  is_train, set_t=-1
    ):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
            #t = torch.randint(0, 1, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
        noisy_data = (1 - cond_mask) * noisy_data
        predicted = self.diffmodel(coeffs, cond_mask, noisy_data, t)

        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        #loss = (torch.abs(residual)).sum() / (num_eval if num_eval > 0 else 1)
        return loss

   
    def impute(self, observed_data, cond_mask,  n_samples, coeffs):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
       

            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                xt = (1-cond_mask)*current_sample
       
                predicted = self.diffmodel(coeffs, cond_mask, xt,  torch.tensor([t]).to(self.device))

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples
    
    def DDIMimpute(self, observed_data, cond_mask,  n_samples, coeffs,eta=1):

        # #先密集后疏松
        DDIM_seq1 = [99, 77, 48, 35, 34, 28, 7]
        #DDIM_seq1 = [49, 42, 41, 10]
        DDIM_seq2 = DDIM_seq1[1:]
        DDIM_seq2.append(0)
    
        

        B, K, L = observed_data.shape
        #noise_a = torch.randn_like(observed_data)   
        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)


        for i in range(n_samples):     
            current_sample = torch.randn_like(observed_data)      
            #current_sample = noise_a  
            scale = [1,1,1,1,1,1,1,1,1,1,1] 
            #scale = [0,0,0,0,0,0,0,0,0,0,0,0]
            for s in range(0,len(DDIM_seq1)):
                t = DDIM_seq1[s]
                t_minus1 = DDIM_seq2[s]
                    
                if s == -1:
                    predicted = current_sample
                    X0_t =  (current_sample-predicted*((1-self.alpha_torch[t])**0.5))/(self.alpha_torch[t]**0.5)
                else:
                    diff_input = ((1-cond_mask)*current_sample) #(B,K,L)
                    predicted = self.diffmodel(coeffs,cond_mask,diff_input,torch.tensor([t]).to(self.device))  
                    X0_t = (current_sample-predicted*((1-self.alpha_torch[t])**0.5))/(self.alpha_torch[t]**0.5)
                ### eta 暂时取0
                # if t_minus1 > 0:
                eta_star = (1 - self.alpha_torch[t_minus1])**0.5/((1-(self.alpha_torch[t]/self.alpha_torch[t_minus1]))*((1-self.alpha_torch[t_minus1])/(1-self.alpha_torch[t])))**0.5
                eta = eta_star*scale[s]
                c1 = (
                        eta * ((1-(self.alpha_torch[t]/self.alpha_torch[t_minus1]))*((1-self.alpha_torch[t_minus1])/(1-self.alpha_torch[t])))**0.5
                )
                c2 = max(0,((1 - self.alpha_torch[t_minus1]) - c1**2)**0.5)
                noise = torch.randn_like(current_sample)
                x_tminus1 = (self.alpha_torch[t_minus1]**0.5)*X0_t + c2*predicted + c1*noise
                c2 = (1 - self.alpha_torch[t_minus1])**0.5
                # x_tminus1_eta0 = (self.alpha_torch[t_minus1]**0.5)*X0_t  + c2*predicted
                # eval_mask = observed_mask-cond_mask
                # print(f"第{t_minus1}步随机误差:",((x_tminus1-observed_data)**2*eval_mask).sum()/eval_mask.sum())
                # print(f"第{t_minus1}步确定误差:",((x_tminus1_eta0-observed_data)**2*eval_mask).sum()/eval_mask.sum())
                #really important
                #current_sample = x_tminus1
                if t_minus1 > 25 :
                    current_sample = x_tminus1
                # else:
                #     c2 = ((1 - self.alpha_torch[t_minus1]) - c1**2)**0.5
                #     current_sample = (self.alpha_torch[t_minus1]**0.5)*X0_t + c2*predicted
                else :
                    current_sample = X0_t
            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
            coeffs,
            cond_mask,
            freq_coeffs,
        ) = self.process_data(batch)

       

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        output = loss_func(observed_data, cond_mask, observed_mask, freq_coeffs, is_train)
        return output

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
            coeffs,
            _,
            freq_coeffs,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

           

            samples = self.impute(observed_data, cond_mask, n_samples, freq_coeffs)
            #samples = self.DDIMimpute(observed_data, cond_mask, n_samples, freq_coeffs)

            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp






class PriSTI_aqi36(PriSTI):
    def __init__(self, config, device, target_dim=36, seq_len=24):
        super(PriSTI_aqi36, self).__init__(target_dim, seq_len, config, device)
        self.config = config

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device).float()
       
        coeffs = None
        if True:
            coeffs = batch["coeffs"].to(self.device).float()
        cond_mask = batch["cond_mask"].to(self.device).float()
        freq_coeffs = batch["freq"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)  # [B, K, L]
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)
        cond_mask = cond_mask.permute(0, 2, 1)
       

        if True:
            coeffs = coeffs.permute(0, 2, 1)
        freq_coeffs = freq_coeffs.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            coeffs,
            cond_mask,
            freq_coeffs,
        )


class PriSTI_MetrLA(PriSTI):
    def __init__(self, config, device, target_dim=207, seq_len=24):
        super(PriSTI_MetrLA, self).__init__(target_dim, seq_len, config, device)
        self.config = config

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        coeffs = None
        if self.config['model']['use_guide']:
            coeffs = batch["coeffs"].to(self.device).float()
        cond_mask = batch["cond_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)  # [B, K, L]
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        cond_mask = cond_mask.permute(0, 2, 1)
        for_pattern_mask = observed_mask

        if self.config['model']['use_guide']:
            coeffs = coeffs.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            coeffs,
            cond_mask,
        )


class PriSTI_PemsBAY(PriSTI):
    def __init__(self, config, device, target_dim=325, seq_len=24):
        super(PriSTI_PemsBAY, self).__init__(target_dim, seq_len, config, device)
        self.config = config

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        coeffs = None
        if self.config['model']['use_guide']:
            coeffs = batch["coeffs"].to(self.device).float()
        cond_mask = batch["cond_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)  # [B, K, L]
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        cond_mask = cond_mask.permute(0, 2, 1)
        for_pattern_mask = observed_mask

        if self.config['model']['use_guide']:
            coeffs = coeffs.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            coeffs,
            cond_mask,
        )

