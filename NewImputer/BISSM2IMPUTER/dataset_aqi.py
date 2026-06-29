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
from utils import get_randmask, get_hist_mask,compute_information_richness,rank_elements



class AQI36_Dataset(Dataset):
    def __init__(self, eval_length=36, target_dim=36, mode="train", val_len=0.1, is_interpolate=False,
                 target_strategy='hybrid', mask_sensor=None, missing_ratio=None):
        self.eval_length = eval_length
        self.target_dim = target_dim
        self.is_interpolate = is_interpolate
        self.target_strategy = target_strategy
        self.mode = mode
        self.missing_ratio = missing_ratio
        self.mask_sensor = mask_sensor
       

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
        if self.is_interpolate:
            tmp_data = torch.tensor(ob_data).to(torch.float64)
            itp_data = torch.where(cond_mask == 0, float('nan'), tmp_data).to(torch.float32)
            itp_data = torchcde.linear_interpolation_coeffs(
                itp_data.permute(1, 0).unsqueeze(-1)).squeeze(-1).permute(1, 0)
            s["coeffs"] = itp_data.numpy()
            ###### for cubic spline interpolation ########
            # tmp_data = torch.tensor(ob_data).to(torch.float64)
            # itp_data = torch.where(cond_mask == 0, float('nan'), tmp_data).to(torch.float32)
            # coeffs = torchcde.natural_cubic_spline_coeffs(itp_data.permute(1,0).unsqueeze(-1))
            # # 创建 CubicSpline 对象
            # Cubic_spline = torchcde.CubicSpline(coeffs)
            # time_points = torch.arange(tmp_data.shape[0], dtype=torch.float32)

            # # 准备存储插值结果的张量
            # interpolated_data = torch.zeros_like(tmp_data)

            # # 对于每个特征 k 和时间点 l，进行插值
            # for k in range(ob_data.shape[1]):  # 对于每个特征
            #     for l in range(ob_data.shape[0]):  # 对于每个时间点
            #         if torch.isnan(itp_data[l, k]):  # 如果当前时间点是缺失值
            #             # 获取当前时间点的时间 t
            #             t = time_points[l]
            #             # 计算插值： f(t) = a + b*t + c*t^2 + d*t^3
            #             interpolated_value = Cubic_spline.evaluate(t)[k]  # 使用 evaluate 方法获取插值结果

            #             # 将插值结果存储到 interpolated_data 中
            #             interpolated_data[l, k] = interpolated_value
            #         else:
            #             # 对于非缺失值，直接使用原始数据
            #             interpolated_data[l, k] = ob_data[l, k]
            # s["coeffs"]=interpolated_data.numpy()

            freq_data = torch.fft.rfft(itp_data, dim=0)  # [L//2 + 1, K]，复数张量

            # 步骤 3：频率滤波
            freq_len = freq_data.shape[0]

            # n_low = int(freq_len * 0.1)
            # freq_data[:n_low, :] = 0
            
            n_high = int(freq_len * 0.2)
            freq_data[-n_high:, :] = 0

            # 步骤 4：逆傅里叶变换
            filtered_data = torch.fft.irfft(freq_data, n=itp_data.shape[0], dim=0).cpu().numpy()  # [L, K]
            filtered_data = filtered_data * (1-cond_mask).cpu().numpy() + ob_data * cond_mask.cpu().numpy()
            
            s["freq"] = filtered_data
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

def get_dataloader(batch_size, device, val_len=0.1, is_interpolate=False, num_workers=4, target_strategy='hybrid', mask_sensor=None):
    dataset = AQI36_Dataset(mode="train", is_interpolate=is_interpolate, target_strategy=target_strategy, mask_sensor=mask_sensor)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    dataset_test = AQI36_Dataset(mode="test", is_interpolate=is_interpolate, target_strategy=target_strategy, mask_sensor=mask_sensor)
    test_loader = DataLoader(
        dataset_test, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    dataset_valid = AQI36_Dataset(mode="valid", val_len=val_len, is_interpolate=is_interpolate, target_strategy=target_strategy, mask_sensor=mask_sensor)
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
 
    