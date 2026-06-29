import sys
from pathlib import Path
import os 
from multiprocessing import Pool, cpu_count


# 将父目录添加到 sys.path 中
sys.path.append("/home/duanlei/PriSTI")


import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
import torchcde
import pywt
from scipy.fft import dct as scipy_dct, idct as scipy_idct
from utils import get_randmask, get_hist_mask,compute_information_richness,rank_elements



class AQI36_Dataset(Dataset):
    def __init__(self, eval_length=36, target_dim=36, mode="train", val_len=0.1, is_interpolate=False,
                 target_strategy='hybrid', mask_sensor=None, missing_ratio=None, preimpute_flag="Linear", freq_flag=True,
                 freq_ratio=0.1, mavg_window=0, decay_alpha=0.0, dct_ratio=0.0, wavelet_level=0,
                 aux_freq=False, aux_freq_ratio=0.1):
        self.eval_length = eval_length
        self.target_dim = target_dim
        self.is_interpolate = is_interpolate
        self.target_strategy = target_strategy
        self.mode = mode
        self.missing_ratio = missing_ratio
        self.mask_sensor = mask_sensor
        self.preimpute_flag = preimpute_flag
        self.freq_flag = freq_flag
        self.freq_ratio = freq_ratio
        self.mavg_window = mavg_window
        self.decay_alpha = decay_alpha
        self.dct_ratio = dct_ratio
        self.wavelet_level = wavelet_level
        self.aux_freq = aux_freq
        self.aux_freq_ratio = aux_freq_ratio

        # 加载邻接矩阵用于空间预填充
        _latlng = pd.read_csv("/home/duanlei/PriSTI/data/pm25/SampleData/pm25_latlng.txt")
        _coords = np.radians(_latlng[['latitude', 'longitude']].values)
        from sklearn.metrics.pairwise import haversine_distances
        _dist = haversine_distances(_coords) * 6371.0088
        _theta = np.std(_dist)
        adj_np = np.exp(-np.square(_dist / _theta))
        adj_np[adj_np < 0.1] = 0.0
        np.fill_diagonal(adj_np, 0.0)
        self.adj = torch.from_numpy(adj_np.astype(np.float32))

        path = "/home/duanlei/PriSTI/data/pm25/pm25_meanstd.pk"
        with open(path, "rb") as f:
            self.train_mean, self.train_std = pickle.load(f)
        if mode == "train":
            month_list = [1, 2, 4, 5, 7, 8, 10, 11]
            # 1st,4th,7th,10th months are excluded from histmask (since the months are used for creating missing patterns in test dataset)
            flag_for_histmask = [0, 1, 0, 1, 0, 1, 0, 1]
        elif mode == "valid":
            month_list = [2, 5, 8, 11]
        elif mode == "test":
            month_list = [3, 6, 9, 12]
        self.month_list = month_list

        # create data for batch
        self.timeofday = []
        self.dayofweek = []
        
        self.observed_data = []  # values (separated into each month)
        self.observed_mask = []  # masks (separated into each month)
        self.gt_mask = []  # ground-truth masks (separated into each month)
        self.index_month = []  # indicate month
        self.position_in_month = []  # indicate the start position in month (length is the same as index_month)
        self.valid_for_histmask = []  # whether the sample is used for histmask
        self.use_index = []  # to separate train/valid/test
        self.cut_length = []  # excluded from evaluation targets

        df = pd.read_csv(
            "/home/duanlei/PriSTI/data/pm25/SampleData/pm25_ground.txt",
            index_col="datetime",
            parse_dates=True,
        )
        self.seasonal_prior = self._build_seasonal_prior(df)
        # 生成星期几 DataFrame
        day_of_week_df = df.index.to_series().apply(lambda x: x.dayofweek).to_frame(name='dow')
        day_of_week_df = pd.concat([day_of_week_df] * df.shape[1], axis=1)
        day_of_week_df.columns = df.columns

        # 生成小时 DataFrame
        hour_of_day_df = df.index.to_series().apply(lambda x: x.hour).to_frame(name='hour')
        hour_of_day_df = pd.concat([hour_of_day_df] * df.shape[1], axis=1)
        hour_of_day_df.columns = df.columns
        df_gt = pd.read_csv(
            "/home/duanlei/PriSTI/data/pm25/SampleData/pm25_missing.txt",
            index_col="datetime",
            parse_dates=True,
        )

        for i in range(len(month_list)):
            current_df = df[df.index.month == month_list[i]]
            current_df_gt = df_gt[df_gt.index.month == month_list[i]]
            current_tod_df = hour_of_day_df[df_gt.index.month == month_list[i]]
            current_dow_df = day_of_week_df[df_gt.index.month == month_list[i]]
            if mode == 'train' and month_list[i] in [2, 5, 8, 11]:
                cut_len = int(val_len * len(current_df))
                current_df = current_df[:-cut_len]
                current_df_gt = current_df_gt[:-cut_len]
            if mode == 'valid':
                cut_len = int(val_len * len(current_df))
                current_df = current_df[-cut_len:]
                current_df_gt = current_df_gt[-cut_len:]
            current_length = len(current_df) - eval_length + 1

            last_index = len(self.index_month)
            self.index_month += np.array([i] * current_length).tolist()
            self.position_in_month += np.arange(current_length).tolist()
            if mode == "train":
                self.valid_for_histmask += np.array(
                    [flag_for_histmask[i]] * current_length
                ).tolist()

            # mask values for observed indices are 1
            c_mask = 1 - current_df.isnull().values
            c_gt_mask = 1 - current_df_gt.isnull().values
            c_tod = current_tod_df.values
            c_dow = current_dow_df.values
            if len(self.mask_sensor) > 0:
                for sensor in self.mask_sensor:
                    c_gt_mask[:, sensor] = 0
                if self.mode == 'train':
                    for sensor in self.mask_sensor:
                        c_mask[:, sensor] = 0
            c_data = (
                (current_df.fillna(0).values - self.train_mean) / self.train_std
            ) * c_mask
            self.observed_mask.append(c_mask)
            self.gt_mask.append(c_gt_mask)
            self.observed_data.append(c_data)
            self.timeofday.append(c_tod)
            self.dayofweek.append(c_dow)

            if mode == "test":
                n_sample = len(current_df) // eval_length
                # interval size is eval_length (missing values are imputed only once)
                c_index = np.arange(
                    last_index, last_index + eval_length * n_sample, eval_length
                )
                self.use_index += c_index.tolist()
                self.cut_length += [0] * len(c_index)
                if len(current_df) % eval_length != 0:  # avoid double-count for the last time-series
                    self.use_index += [len(self.index_month) - 1]
                    self.cut_length += [eval_length - len(current_df) % eval_length]

        if mode != "test":
            self.use_index = np.arange(len(self.index_month))
            self.cut_length = [0] * len(self.use_index)

        # masks for 1st,4th,7th,10th months are used for creating missing patterns in test data,
        # so these months are excluded from histmask to avoid leakage
        if mode == "train":
            ind = -1
            self.index_month_histmask = []
            self.position_in_month_histmask = []

            for i in range(len(self.index_month)):
                while True:
                    ind += 1
                    if ind == len(self.index_month):
                        ind = 0
                    if self.valid_for_histmask[ind] == 1:
                        self.index_month_histmask.append(self.index_month[ind])
                        self.position_in_month_histmask.append(
                            self.position_in_month[ind]
                        )
                        break
        else:  # dummy (histmask is only used for training)
            self.index_month_histmask = self.index_month
            self.position_in_month_histmask = self.position_in_month

    def _build_seasonal_prior(self, df):
        train_months = [1, 2, 4, 5, 7, 8, 10, 11]
        seasonal_df = df[df.index.month.isin(train_months)].copy()
        seasonal_df = (seasonal_df - self.train_mean) / self.train_std
        hour_mean = seasonal_df.groupby(seasonal_df.index.hour).mean()
        hour_mean = hour_mean.reindex(range(24)).ffill().bfill()
        dow_hour_mean = seasonal_df.groupby([seasonal_df.index.dayofweek, seasonal_df.index.hour]).mean()

        seasonal_prior = np.zeros((7, 24, self.target_dim), dtype=np.float32)
        for dow in range(7):
            for hour in range(24):
                if (dow, hour) in dow_hour_mean.index:
                    values = dow_hour_mean.loc[(dow, hour)].to_numpy(dtype=np.float32)
                else:
                    values = hour_mean.loc[hour].to_numpy(dtype=np.float32)
                fallback = hour_mean.loc[hour].to_numpy(dtype=np.float32)
                values = np.where(np.isnan(values), fallback, values)
                seasonal_prior[dow, hour] = values
        return seasonal_prior

    def _compute_directional_fills(self, ob_data, cond_mask):
        tmp_data = torch.tensor(ob_data).to(torch.float32)
        L_t, K_t = tmp_data.shape
        large_dist = float(L_t + 1)

        fwd = tmp_data.clone()
        bwd = tmp_data.clone()
        fwd_dist = torch.full((L_t, K_t), large_dist, dtype=torch.float32)
        bwd_dist = torch.full((L_t, K_t), large_dist, dtype=torch.float32)

        fwd_dist[0] = torch.where(
            cond_mask[0] == 1,
            torch.zeros(K_t, dtype=torch.float32),
            torch.full((K_t,), large_dist, dtype=torch.float32),
        )
        for t in range(1, L_t):
            fwd[t] = torch.where(cond_mask[t] == 0, fwd[t - 1], fwd[t])
            fwd_dist[t] = torch.where(
                cond_mask[t] == 1,
                torch.zeros(K_t, dtype=torch.float32),
                torch.where(cond_mask[t - 1] == 1, torch.ones(K_t, dtype=torch.float32), fwd_dist[t - 1] + 1),
            )

        bwd_dist[-1] = torch.where(
            cond_mask[-1] == 1,
            torch.zeros(K_t, dtype=torch.float32),
            torch.full((K_t,), large_dist, dtype=torch.float32),
        )
        for t in range(L_t - 2, -1, -1):
            bwd[t] = torch.where(cond_mask[t] == 0, bwd[t + 1], bwd[t])
            bwd_dist[t] = torch.where(
                cond_mask[t] == 1,
                torch.zeros(K_t, dtype=torch.float32),
                torch.where(cond_mask[t + 1] == 1, torch.ones(K_t, dtype=torch.float32), bwd_dist[t + 1] + 1),
            )

        return tmp_data, fwd, bwd, fwd_dist, bwd_dist, large_dist

    def _get_seasonal_prior(self, time_of_day, day_of_week):
        tod_index = time_of_day[:, 0].astype(np.int64)
        dow_index = day_of_week[:, 0].astype(np.int64)
        return torch.from_numpy(self.seasonal_prior[dow_index, tod_index]).to(torch.float32)

    def _seasonal_bridge_fill(self, ob_data, cond_mask, time_of_day, day_of_week):
        tmp_data, fwd, bwd, fwd_dist, bwd_dist, large_dist = self._compute_directional_fills(ob_data, cond_mask)
        prior = self._get_seasonal_prior(time_of_day, day_of_week)

        has_left = fwd_dist < large_dist
        has_right = bwd_dist < large_dist
        has_both = has_left & has_right
        total_dist = (fwd_dist + bwd_dist).clamp_min(1.0)
        linear_bridge = (fwd * bwd_dist + bwd * fwd_dist) / total_dist

        gap_len = torch.where(
            has_both,
            (fwd_dist + bwd_dist - 1.0).clamp_min(1.0),
            torch.where(has_left, fwd_dist.clamp_min(1.0), bwd_dist.clamp_min(1.0)),
        )

        forward_weight = ((6.0 - gap_len) / 5.0).clamp(0.35, 0.9)
        base_fill = forward_weight * fwd + (1.0 - forward_weight) * linear_bridge
        base_fill = torch.where(
            has_both,
            base_fill,
            torch.where(has_left, fwd, torch.where(has_right, bwd, prior)),
        )

        prior_weight = ((gap_len - 3.0) / 10.0).clamp(0.0, 0.25)
        prior_weight = torch.where(has_left ^ has_right, torch.maximum(prior_weight, torch.full_like(prior_weight, 0.15)), prior_weight)
        itp_data = (1.0 - prior_weight) * base_fill + prior_weight * prior
        itp_data = torch.where(cond_mask == 1, tmp_data, itp_data)
        return itp_data



    def __getitem__(self, org_index):
        #print(self.use_index)
        index = self.use_index[org_index]
        c_month = self.index_month[index]
        c_index = self.position_in_month[index]

        index2 = np.random.randint(0, len(self.use_index))
        hist_month = self.index_month_histmask[index2]
        hist_index = self.position_in_month_histmask[index2]

        ob_data = self.observed_data[c_month][c_index:c_index + self.eval_length]
        ob_mask = self.observed_mask[c_month][c_index:c_index + self.eval_length]
        ob_mask_t = torch.tensor(ob_mask).float()
        gt_mask = self.gt_mask[c_month][c_index:c_index + self.eval_length]
        for_pattern_mask = self.observed_mask[hist_month][hist_index:hist_index + self.eval_length]

        if self.mode != 'train':
            cond_mask = torch.tensor(gt_mask).to(torch.float32)
        else:
            if self.target_strategy != 'random':
                cond_mask = get_hist_mask(ob_mask_t, for_pattern_mask=for_pattern_mask)
            else:
                cond_mask = get_randmask(ob_mask_t)
        
        time_richness,space_richness = compute_information_richness(cond_mask)
        total_richness = time_richness+0.01*space_richness
        rank_of_cond_mask = rank_elements(total_richness,cond_mask)
        #self.validate_rank_condition(cond_mask,rank_of_cond_mask)
  
        # current_cond_mask = self.gen_mask(cond_mask,rank_of_cond_mask,propotion=1.0)
        # print(current_cond_mask)
        s = {
            "observed_data": ob_data,
            "observed_mask": ob_mask,
            "gt_mask": gt_mask,
            "hist_mask": for_pattern_mask,
            "timepoints": np.arange(self.eval_length),
            "cut_length": self.cut_length[org_index],
            "cond_mask": cond_mask.numpy(),
            "rank_of_cond_mask":rank_of_cond_mask,
            "time_of_day":self.timeofday[c_month][
                c_index : c_index + self.eval_length
            ],
            "day_of_week":self.dayofweek[c_month][
                c_index : c_index + self.eval_length
            ],
        }
        preimpute_flag = self.preimpute_flag
        freq_flag = self.freq_flag
        if self.is_interpolate:
            if preimpute_flag == "Linear":
                tmp_data = torch.tensor(ob_data).to(torch.float32)
                itp_data = torch.where(cond_mask == 0, float('nan'), tmp_data).to(torch.float32)
                itp_data = torchcde.linear_interpolation_coeffs(
                    itp_data.permute(1, 0).unsqueeze(-1)).squeeze(-1).permute(1, 0)
                if freq_flag == False:
                    s["coeffs"] = itp_data.numpy()
###################################################################################
            elif preimpute_flag == "Forward":
                tmp_data = torch.tensor(ob_data).to(torch.float32)
                # 使用前一时刻的值进行插补
                for t in range(1, tmp_data.shape[0]):
                    tmp_data[t] = torch.where(cond_mask[t] == 0, tmp_data[t - 1], tmp_data[t])
                itp_data = tmp_data.clone()
                if freq_flag == False:
                    s["coeffs"] = itp_data.numpy()
####################################################################################
            elif preimpute_flag == "Backward":
                tmp_data = torch.tensor(ob_data).to(torch.float32)
                # 使用后一时刻的值进行后向插补
                for t in range(tmp_data.shape[0] - 2, -1, -1):
                    tmp_data[t] = torch.where(cond_mask[t] == 0, tmp_data[t + 1], tmp_data[t])
                itp_data = tmp_data.clone()
                if freq_flag == False:
                    s["coeffs"] = itp_data.numpy()   
            elif preimpute_flag == "Bidirectional":
                tmp_data = torch.tensor(ob_data).to(torch.float32)
                # Forward fill
                fwd = tmp_data.clone()
                for t in range(1, fwd.shape[0]):
                    fwd[t] = torch.where(cond_mask[t] == 0, fwd[t - 1], fwd[t])
                # Backward fill
                bwd = tmp_data.clone()
                for t in range(bwd.shape[0] - 2, -1, -1):
                    bwd[t] = torch.where(cond_mask[t] == 0, bwd[t + 1], bwd[t])
                # 取均值
                itp_data = (fwd + bwd) / 2.0
                # 保留观测值
                itp_data = torch.where(cond_mask == 1, tmp_data, itp_data)
                if freq_flag == False:
                    s["coeffs"] = itp_data.numpy()
            elif preimpute_flag == "ExpDecay":
                tmp_data = torch.tensor(ob_data).to(torch.float32)
                alpha = self.decay_alpha
                # Forward fill with exponential decay toward 0 (normalized mean)
                dist = torch.zeros(tmp_data.shape[1])  # distance since last known, per feature
                for t in range(1, tmp_data.shape[0]):
                    dist = torch.where(cond_mask[t - 1] == 1, torch.ones_like(dist), dist + 1)
                    decay = torch.exp(-alpha * dist)
                    tmp_data[t] = torch.where(cond_mask[t] == 0, tmp_data[t - 1] * decay, tmp_data[t])
                itp_data = tmp_data.clone()
                if freq_flag == False:
                    s["coeffs"] = itp_data.numpy()
            elif preimpute_flag == "DistWeighted":
                # 距离加权双向填充：按到最近已知值的距离反比加权
                tmp_data = torch.tensor(ob_data).to(torch.float32)
                L_t, K_t = tmp_data.shape
                fwd = tmp_data.clone()
                fwd_dist = torch.zeros(L_t, K_t)  # 与前向最近已知值的距离
                for t in range(1, L_t):
                    fwd[t] = torch.where(cond_mask[t] == 0, fwd[t - 1], fwd[t])
                    fwd_dist[t] = torch.where(cond_mask[t] == 1, torch.zeros(K_t), fwd_dist[t - 1] + 1)
                bwd = tmp_data.clone()
                bwd_dist = torch.zeros(L_t, K_t)
                for t in range(L_t - 2, -1, -1):
                    bwd[t] = torch.where(cond_mask[t] == 0, bwd[t + 1], bwd[t])
                    bwd_dist[t] = torch.where(cond_mask[t] == 1, torch.zeros(K_t), bwd_dist[t + 1] + 1)
                # 距离越近权重越大，避免除零
                fwd_w = 1.0 / (fwd_dist + 1.0)
                bwd_w = 1.0 / (bwd_dist + 1.0)
                total_w = fwd_w + bwd_w
                itp_data = (fwd * fwd_w + bwd * bwd_w) / total_w
                itp_data = torch.where(cond_mask == 1, tmp_data, itp_data)
                if freq_flag == False:
                    s["coeffs"] = itp_data.numpy()
            elif preimpute_flag == "NoisyForward":
                # Forward fill + 高斯噪声扰动（训练时加噪声增强鲁棒性）
                tmp_data = torch.tensor(ob_data).to(torch.float32)
                for t in range(1, tmp_data.shape[0]):
                    tmp_data[t] = torch.where(cond_mask[t] == 0, tmp_data[t - 1], tmp_data[t])
                itp_data = tmp_data.clone()
                if self.mode == "train":
                    noise_scale = self.decay_alpha if self.decay_alpha > 0 else 0.1
                    noise = torch.randn_like(itp_data) * noise_scale
                    miss = (1.0 - cond_mask)
                    itp_data = itp_data + noise * miss
                if freq_flag == False:
                    s["coeffs"] = itp_data.numpy()
            elif preimpute_flag == "SeasonalBridge":
                itp_data = self._seasonal_bridge_fill(
                    ob_data,
                    cond_mask,
                    self.timeofday[c_month][c_index:c_index + self.eval_length],
                    self.dayofweek[c_month][c_index:c_index + self.eval_length],
                )
                if freq_flag == False:
                    s["coeffs"] = itp_data.numpy()
            elif preimpute_flag.startswith("Blend"):
                # α-Blend: fill = α·Forward + (1-α)·Linear
                alpha = int(preimpute_flag.replace("Blend", "")) / 10.0
                # Forward fill
                tmp_fwd = torch.tensor(ob_data).to(torch.float32)
                for t in range(1, tmp_fwd.shape[0]):
                    tmp_fwd[t] = torch.where(cond_mask[t] == 0, tmp_fwd[t - 1], tmp_fwd[t])
                # Linear interpolation
                tmp_lin = torch.tensor(ob_data).to(torch.float32)
                tmp_lin = torch.where(cond_mask == 0, float('nan'), tmp_lin)
                tmp_lin = torchcde.linear_interpolation_coeffs(
                    tmp_lin.permute(1, 0).unsqueeze(-1)).squeeze(-1).permute(1, 0)
                # Blend
                itp_data = alpha * tmp_fwd + (1 - alpha) * tmp_lin
                itp_data = torch.where(cond_mask == 1, torch.tensor(ob_data).float(), itp_data)
                if freq_flag == False:
                    s["coeffs"] = itp_data.numpy()
            elif preimpute_flag == "Spatial":
                # 空间加权预填充：用邻居观测值的邻接权重加权平均
                tmp_data = torch.tensor(ob_data).to(torch.float32)
                adj = self.adj  # (K, K)
                L_t, K_t = tmp_data.shape
                itp_data = tmp_data.clone()
                cond_f = cond_mask.float()
                for t in range(L_t):
                    obs_t = cond_f[t]  # (K,) 1=observed
                    miss_idx = (obs_t == 0).nonzero(as_tuple=True)[0]
                    if miss_idx.numel() == 0:
                        continue
                    # 邻居权重 = adj[miss, :] * obs_mask (只取已观测邻居)
                    w = adj[miss_idx] * obs_t.unsqueeze(0)  # (n_miss, K)
                    w_sum = w.sum(dim=1, keepdim=True).clamp_min(1e-8)
                    spatial_fill = (w @ tmp_data[t].unsqueeze(1)).squeeze(1) / w_sum.squeeze(1)
                    itp_data[t, miss_idx] = spatial_fill
                # 如果仍有缺失（所有邻居也缺失），退回前向填充
                still_miss = (cond_mask == 0) & (itp_data == 0)
                if still_miss.any():
                    for t in range(1, L_t):
                        itp_data[t] = torch.where(
                            (cond_mask[t] == 0) & (itp_data[t] == 0),
                            itp_data[t - 1], itp_data[t])
                if freq_flag == False:
                    s["coeffs"] = itp_data.numpy()
            elif preimpute_flag == "SpatialForward":
                # 先前向填充，再对缺失位置用 0.5*spatial + 0.5*forward 混合
                tmp_data = torch.tensor(ob_data).to(torch.float32)
                adj = self.adj
                L_t, K_t = tmp_data.shape
                cond_f = cond_mask.float()
                # Forward fill
                fwd = tmp_data.clone()
                for t in range(1, L_t):
                    fwd[t] = torch.where(cond_f[t] == 0, fwd[t - 1], fwd[t])
                # Spatial fill
                spatial = tmp_data.clone()
                for t in range(L_t):
                    obs_t = cond_f[t]
                    miss_idx = (obs_t == 0).nonzero(as_tuple=True)[0]
                    if miss_idx.numel() == 0:
                        continue
                    w = adj[miss_idx] * obs_t.unsqueeze(0)
                    w_sum = w.sum(dim=1, keepdim=True).clamp_min(1e-8)
                    spatial[t, miss_idx] = (w @ tmp_data[t].unsqueeze(1)).squeeze(1) / w_sum.squeeze(1)
                # 有空间信息的位置用 0.5 混合，无空间信息的纯用 forward
                has_spatial = (spatial != tmp_data) | (cond_f == 1)
                itp_data = torch.where(
                    (cond_f == 0) & has_spatial,
                    0.5 * spatial + 0.5 * fwd,
                    fwd
                )
                itp_data = torch.where(cond_f == 1, tmp_data, itp_data)
                if freq_flag == False:
                    s["coeffs"] = itp_data.numpy()
            elif preimpute_flag == "ForwardLinear":
                # 双通道：Forward for main, Linear for aux (freq_coeffs)
                tmp_fwd = torch.tensor(ob_data).to(torch.float32)
                for t in range(1, tmp_fwd.shape[0]):
                    tmp_fwd[t] = torch.where(cond_mask[t] == 0, tmp_fwd[t - 1], tmp_fwd[t])
                itp_data = tmp_fwd.clone()
                s["coeffs"] = itp_data.numpy()
                # Linear interpolation as aux channel
                tmp_lin = torch.tensor(ob_data).to(torch.float32)
                tmp_lin = torch.where(cond_mask == 0, float('nan'), tmp_lin)
                tmp_lin = torchcde.linear_interpolation_coeffs(
                    tmp_lin.permute(1, 0).unsqueeze(-1)).squeeze(-1).permute(1, 0)
                lin_np = tmp_lin.numpy()
                lin_np = lin_np * (1 - cond_mask).cpu().numpy() + ob_data * cond_mask.cpu().numpy()
                s["freq_coeffs"] = lin_np
            else: 
                itp_data = itp_data = ob_data.copy() * cond_mask.clone().numpy()
                s["coeffs"] = itp_data

            if freq_flag == True and preimpute_flag not in ("None", "ForwardLinear"):
                freq_data = torch.fft.rfft(itp_data, dim=0)  # [L//2 + 1, K]，复数张量

                # 频率滤波
                freq_len = freq_data.shape[0]
                n_high = max(1, int(freq_len * self.freq_ratio))
                freq_data[-n_high:, :] = 0

                # 逆傅里叶变换
                filtered_data = torch.fft.irfft(freq_data, n=itp_data.shape[0], dim=0).cpu().numpy()  # [L, K]
                filtered_data = filtered_data * (1-cond_mask).cpu().numpy() + ob_data * cond_mask.cpu().numpy()
                
                s["coeffs"] = filtered_data

            elif self.mavg_window > 0 and preimpute_flag != "None":
                # 滑动窗口平滑
                if isinstance(itp_data, torch.Tensor):
                    itp_np = itp_data.numpy()
                else:
                    itp_np = itp_data.copy()
                L, K = itp_np.shape
                w = self.mavg_window
                padded = np.pad(itp_np, ((w//2, w//2), (0, 0)), mode='edge')
                smoothed = np.zeros_like(itp_np)
                for i in range(L):
                    smoothed[i] = padded[i:i+w].mean(axis=0)
                smoothed = smoothed * (1-cond_mask).cpu().numpy() + ob_data * cond_mask.cpu().numpy()
                s["coeffs"] = smoothed

            elif self.dct_ratio > 0 and preimpute_flag != "None":
                # DCT 滤波（替代 FFT，无 Gibbs 现象）
                if isinstance(itp_data, torch.Tensor):
                    itp_np = itp_data.numpy()
                else:
                    itp_np = itp_data.copy()
                dct_data = scipy_dct(itp_np, axis=0, type=2, norm='ortho')
                n_keep = max(1, int(dct_data.shape[0] * (1 - self.dct_ratio)))
                dct_data[n_keep:, :] = 0
                filtered = scipy_idct(dct_data, axis=0, type=2, norm='ortho')
                filtered = filtered * (1-cond_mask).cpu().numpy() + ob_data * cond_mask.cpu().numpy()
                s["coeffs"] = filtered

            elif self.wavelet_level > 0 and preimpute_flag != "None":
                # 小波去噪
                if isinstance(itp_data, torch.Tensor):
                    itp_np = itp_data.numpy()
                else:
                    itp_np = itp_data.copy()
                L, K = itp_np.shape
                smoothed = np.zeros_like(itp_np)
                for k in range(K):
                    coeffs = pywt.wavedec(itp_np[:, k], 'db4', level=self.wavelet_level)
                    # 对细节系数做软阈值去噪
                    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
                    threshold = sigma * np.sqrt(2 * np.log(L))
                    coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
                    rec = pywt.waverec(coeffs, 'db4')
                    smoothed[:, k] = rec[:L]
                smoothed = smoothed * (1-cond_mask).cpu().numpy() + ob_data * cond_mask.cpu().numpy()
                s["coeffs"] = smoothed

            # --- 辅助频率输入：始终提供 raw + FFT滤波 ---
            if self.aux_freq and preimpute_flag not in ("None", "ForwardLinear"):
                # coeffs = raw pre-imputed
                if isinstance(itp_data, torch.Tensor):
                    raw_np = itp_data.numpy()
                else:
                    raw_np = itp_data.copy()
                s["coeffs"] = raw_np  # 覆盖为原始预填充
                # freq_coeffs = FFT 低通滤波
                freq_data = torch.fft.rfft(torch.tensor(raw_np, dtype=torch.float32), dim=0)
                freq_len = freq_data.shape[0]
                n_high = max(1, int(freq_len * self.aux_freq_ratio))
                freq_data[-n_high:, :] = 0
                filtered = torch.fft.irfft(freq_data, n=raw_np.shape[0], dim=0).numpy()
                filtered = filtered * (1 - cond_mask).cpu().numpy() + ob_data * cond_mask.cpu().numpy()
                s["freq_coeffs"] = filtered

        return s
    # def validate_rank_condition(self,cond_mask, rank_of_cond_mask):
    #     """
    #     验证对于每个 batch，cond_mask=0 处的 rank_of_cond_mask 是否都小于 cond_mask=1 处的 rank_of_cond_mask。
    #     Args:
    #         cond_mask: (B, K, L), 二值掩码张量，1 表示已知值，0 表示未知值。
    #         rank_of_cond_mask: (B, K, L), 位次矩阵，表示每个元素在其 (K, L) 矩阵中的排序位次。
    #     Returns:
    #         bool: 如果所有 batch 都满足条件，返回 True；否则返回 False。
    #     """
 


    #     # 获取当前 batch 的 cond_mask 和 rank_of_cond_mask
    #     batch_cond_mask = cond_mask  # (K, L)
    #     batch_rank = rank_of_cond_mask  # (K, L)

    #     # 找到 cond_mask=0 和 cond_mask=1 的位置
    #     zero_indices = (batch_cond_mask == 0)  # cond_mask=0 的位置
    #     one_indices = (batch_cond_mask == 1)   # cond_mask=1 的位置

    #     # 获取 cond_mask=0 和 cond_mask=1 处的 rank_of_cond_mask
    #     rank_zero = batch_rank[zero_indices]  # cond_mask=0 处的 rank
    #     rank_one = batch_rank[one_indices]    # cond_mask=1 处的 rank

    #     # 检查 cond_mask=0 处的 rank 是否都小于 cond_mask=1 处的 rank
    #     if not (rank_zero.max() < rank_one.min()):
    #         print(f" 不满足条件！")
    #         print(f"cond_mask=0 处的 rank: {rank_zero.max()}")
    #         print(f"cond_mask=1 处的 rank: {rank_one.min()}")
    #         return False

    #     print("✅ 所有 batch 都满足条件！")
    #     return True
    def __len__(self):
        return len(self.use_index)
    def gen_mask(self,cond_mask, rank_elements, propotion):
        """
        生成新的 cond_mask，根据 rank_elements 和 propotion 动态调整。
        Args:
            cond_mask: (B, K, L), 二值掩码张量，1 表示已知值，0 表示未知值。
            rank_elements: (B, K, L), 位次矩阵，表示每个元素在其 (K, L) 矩阵中的排序位次。
            propotion: (B,), 每个 batch 的比例，表示需要设置为 1 的未知值比例。
        Returns:
            cond_mask: (B, K, L), 更新后的掩码张量。
        """
        L,K = cond_mask.shape

        # 计算每个 batch 中需要设置为 1 的未知值数量
        num_elements = (cond_mask == 0).sum()  # (B,)
        num_cond = (num_elements * propotion).long()  # (B,)

        # 获取当前 batch 的 rank_elements 和 cond_mask
        batch_rank = rank_elements  # (K, L)
        batch_mask = cond_mask  # (K, L)

        # 找到 rank_elements 中值小于 num_cond[b] 的位置
        selected_indices = (batch_rank < num_cond)

        # 将对应位置的 cond_mask 设置为 1
        batch_mask[selected_indices] = 1

        # 更新 cond_mask
        cond_mask = batch_mask

        return cond_mask

def get_dataloader(batch_size, device, val_len=0.1, is_interpolate=False, num_workers=4, target_strategy='hybrid', mask_sensor=None, preimpute_flag="Linear", freq_flag=True, freq_ratio=0.1, mavg_window=0, decay_alpha=0.0, dct_ratio=0.0, wavelet_level=0, aux_freq=False, aux_freq_ratio=0.1):
    dataset = AQI36_Dataset(mode="train", is_interpolate=is_interpolate, target_strategy=target_strategy, mask_sensor=mask_sensor, preimpute_flag=preimpute_flag, freq_flag=freq_flag, freq_ratio=freq_ratio, mavg_window=mavg_window, decay_alpha=decay_alpha, dct_ratio=dct_ratio, wavelet_level=wavelet_level, aux_freq=aux_freq, aux_freq_ratio=aux_freq_ratio)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    dataset_test = AQI36_Dataset(mode="test", is_interpolate=is_interpolate, target_strategy=target_strategy, mask_sensor=mask_sensor, preimpute_flag=preimpute_flag, freq_flag=freq_flag, freq_ratio=freq_ratio, mavg_window=mavg_window, decay_alpha=decay_alpha, dct_ratio=dct_ratio, wavelet_level=wavelet_level, aux_freq=aux_freq, aux_freq_ratio=aux_freq_ratio)
    test_loader = DataLoader(
        dataset_test, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    dataset_valid = AQI36_Dataset(mode="valid", val_len=val_len, is_interpolate=is_interpolate, target_strategy=target_strategy, mask_sensor=mask_sensor, preimpute_flag=preimpute_flag, freq_flag=freq_flag, freq_ratio=freq_ratio, mavg_window=mavg_window, decay_alpha=decay_alpha, dct_ratio=dct_ratio, wavelet_level=wavelet_level, aux_freq=aux_freq, aux_freq_ratio=aux_freq_ratio)
    valid_loader = DataLoader(
        dataset_valid, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    scaler = torch.from_numpy(dataset.train_std).to(device).float()
    mean_scaler = torch.from_numpy(dataset.train_mean).to(device).float()

    return train_loader, valid_loader, test_loader, scaler, mean_scaler




if __name__=="__main__":
    dataset = AQI36_Dataset(mode="test", is_interpolate=True, target_strategy="hybrid", mask_sensor=[])
    #save_rank_of_cond_mask(dataset, save_dir="/home/duanlei/PriSTI/rank_of_cond_masks", num_workers=40)  # 使用 8 个进程
    for i in range(len(dataset)):
        data = dataset[i]
 
    