import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import logging
import time

def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    foldername="",
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    is_lr_decay = config["is_lr_decay"]
    if foldername != "":
        output_path = foldername + "/model.pth"
        logging.basicConfig(filename=foldername + '/train_model.log', level=logging.DEBUG)
    if is_lr_decay:
        p1 = int(0.75 * config["epochs"])
        p2 = int(0.9 * config["epochs"])
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[p1, p2], gamma=0.1
        )

    valid_epoch_interval = config["valid_epoch_interval"]
    best_valid_loss = 1e10

    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                loss = model(train_batch)
                loss.backward()
                #print(model.revin_layer.affine_weight)
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            logging.info("avg_epoch_loss:" + str(avg_loss / batch_no) + ", epoch:" + str(epoch_no))
            if is_lr_decay:
                lr_scheduler.step()
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0 and (epoch_no + 1) > config["epochs"] * 0.5:
        #if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0 :
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
                    logging.info("valid_avg_epoch_loss"+str(avg_loss_valid / batch_no)+", epoch:"+str(epoch_no))
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )
                logging.info("best loss is updated to "+str(avg_loss_valid / batch_no)+" at "+str(epoch_no))
                if foldername != "":
                    torch.save(model.state_dict(), foldername + "/tmp_model"+str(epoch_no)+".pth")

    if foldername != "":
        torch.save(model.state_dict(), output_path)


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)


def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""):
    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)

                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                # #for revin
                # (
                #     observed_data,
                #     observed_mask,
                #     observed_tp,
                #     gt_mask,
                #     _,
                #     cut_length,
                #     coeffs,
                #     _,
                # ) = model.process_data(test_batch)
                # #model.revin_layer.print_info()
                # result = samples_median.values
                # result = model.revin_layer(result,coeffs.clone().permute(0,2,1),mode='denorm')
                # #model.revin_layer.print_info()
                # mse_current = (
                #     ((result - c_target) * eval_points) ** 2
                # )* (scaler ** 2)
                # mae_current = (
                #     torch.abs((result - c_target) * eval_points) 
                # )* scaler

                mse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points) 
                ) * scaler

                # mse_current = (
                #     ((samples_median.values - c_target) * eval_points) ** 2
                # ) 
                # mae_current = (
                #     torch.abs((samples_median.values - c_target) * eval_points) 
                # ) 
                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )
                logging.info("rmse_total={}".format(np.sqrt(mse_total / evalpoints_total)))
                logging.info("mae_total={}".format(mae_total / evalpoints_total))
                logging.info("batch_no={}".format(batch_no))

            with open(
                foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                all_target = torch.cat(all_target, dim=0)
                all_evalpoint = torch.cat(all_evalpoint, dim=0)
                all_observed_point = torch.cat(all_observed_point, dim=0)
                all_observed_time = torch.cat(all_observed_time, dim=0)
                all_generated_samples = torch.cat(all_generated_samples, dim=0)

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )

            CRPS = calc_quantile_CRPS(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )

            with open(
                foldername + "/result_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                pickle.dump(
                    [
                        np.sqrt(mse_total / evalpoints_total),
                        mae_total / evalpoints_total,
                        CRPS,
                    ],
                    f,
                )
                print("RMSE:", np.sqrt(mse_total / evalpoints_total))
                print("MAE:", mae_total / evalpoints_total)
                print("CRPS:", CRPS)
                logging.info("RMSE={}".format(np.sqrt(mse_total / evalpoints_total)))
                logging.info("MAE={}".format(mae_total / evalpoints_total))
                logging.info("CRPS={}".format(CRPS))


def get_randmask(observed_mask, min_miss_ratio=0., max_miss_ratio=1.):
    rand_for_mask = torch.rand_like(observed_mask) * observed_mask
    rand_for_mask = rand_for_mask.reshape(-1)
    sample_ratio = np.random.rand()
    sample_ratio = sample_ratio * (max_miss_ratio-min_miss_ratio) + min_miss_ratio
    num_observed = observed_mask.sum().item()
    num_masked = round(num_observed * sample_ratio)
    rand_for_mask[rand_for_mask.topk(num_masked).indices] = -1

    cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
    return cond_mask


def get_hist_mask(observed_mask, for_pattern_mask=None, target_strategy='hybrid'):
    if for_pattern_mask is None:
        for_pattern_mask = observed_mask
    if target_strategy == "hybrid":
        rand_mask = get_randmask(observed_mask)

    cond_mask = observed_mask.clone()
    mask_choice = np.random.rand()
    if target_strategy == "hybrid" and mask_choice > 0.5:
        cond_mask = rand_mask
    else:  # draw another sample for histmask (i-1 corresponds to another sample)
        cond_mask = cond_mask * for_pattern_mask
    return cond_mask


def get_block_mask(observed_mask, target_strategy='block'):
    rand_sensor_mask = torch.rand_like(observed_mask)
    randint = np.random.randint
    sample_ratio = np.random.rand()
    sample_ratio = sample_ratio * 0.15
    mask = rand_sensor_mask < sample_ratio
    min_seq = 12
    max_seq = 24
    for col in range(observed_mask.shape[1]):
        idxs = np.flatnonzero(mask[:, col])
        if not len(idxs):
            continue
        fault_len = min_seq
        if max_seq > min_seq:
            fault_len = fault_len + int(randint(max_seq - min_seq))
        idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
        idxs = np.unique(idxs_ext)
        idxs = np.clip(idxs, 0, observed_mask.shape[0] - 1)
        mask[idxs, col] = True
    rand_base_mask = torch.rand_like(observed_mask) < 0.05
    reverse_mask = mask | rand_base_mask
    block_mask = 1 - reverse_mask.to(torch.float32)

    cond_mask = observed_mask.clone()
    mask_choice = np.random.rand()
    if target_strategy == "hybrid" and mask_choice > 0.7:
        cond_mask = get_randmask(observed_mask, 0., 1.)
    else:
        cond_mask = block_mask * cond_mask

    return cond_mask


def compute_information_richness(cond_mask, sigma=1.0):
    L, K = cond_mask.shape
    time_richness = torch.zeros_like(cond_mask, dtype=torch.float32)
    space_richness = torch.zeros_like(cond_mask, dtype=torch.float32)
    sigma = torch.tensor(sigma, dtype=torch.float32)

    # 预计算高斯核
    dist = torch.arange(L, device=cond_mask.device).unsqueeze(1) - torch.arange(L, device=cond_mask.device).unsqueeze(0)  # (L, L)
    gaussian_kernel = torch.exp(-torch.abs(dist).float() ** 2 / (2 * sigma ** 2))  # (L, L)

    # 计算时间丰富度
    time_richness = torch.matmul(gaussian_kernel, cond_mask.float())  # (L, K)
    time_richness[cond_mask != 0] = 0  # 仅保留未知值位置的结果

    # 计算空间丰富度
    num_known_per_time = cond_mask.sum(dim=1, keepdim=True)  # 每个时间点上的已知值数量 (L, 1)
    space_richness = torch.where(cond_mask == 0, num_known_per_time, torch.tensor(0.0))  # 仅对未知值位置赋值

    return time_richness, space_richness

def rank_elements(tensor, cond_mask, descending=True):
    """
    对输入的 (L, K) 或 (B, L, K) 张量进行排序，生成位次矩阵。
    确保：
    1. tensor 中越大的位置，rank 越小。
    2. 无论 tensor 的值如何，cond_mask=1 的位置的 rank 大于 cond_mask=0 的位置的 rank。
    Args:
        tensor: (L, K) 或 (B, L, K), 输入张量。
        cond_mask: (L, K) 或 (B, L, K), 二值掩码张量，1 表示已知值，0 表示未知值。
        descending: bool, 是否按降序排序。
    Returns:
        rank_matrix: (L, K) 或 (B, L, K), 位次矩阵，值表示元素在排序中的位次。
    """
    if tensor.dim() == 2:  # 如果输入是 (L, K)
        L, K = tensor.shape
        flat_tensor = tensor.reshape(-1)  # 展平为一维向量 (L * K,)
        flat_cond_mask = cond_mask.reshape(-1)  # 展平 cond_mask

        # 将 cond_mask=1 的位置的值设置为一个较小的固定值
        flat_tensor[flat_cond_mask == 1] = -1e6

        # 排序
        _, indices = torch.sort(flat_tensor, descending=descending)  # 按降序排序
        rank_matrix = torch.zeros_like(flat_tensor, dtype=torch.float32)  # 初始化位次矩阵
        rank_matrix[indices] = torch.arange(L * K, dtype=torch.float32, device=tensor.device)  # 生成位次
        rank_matrix = rank_matrix.reshape(L, K)  # 恢复为 (L, K) 的形状

    elif tensor.dim() == 3:  # 如果输入是 (B, L, K)
        B, L, K = tensor.shape
        flat_tensor = tensor.reshape(B, -1)  # 展平为 (B, L * K)
        flat_cond_mask = cond_mask.reshape(B, -1)  # 展平 cond_mask

        # 将 cond_mask=1 的位置的值设置为一个较小的固定值
        flat_tensor[flat_cond_mask == 1] = -1e6

        # 排序
        _, indices = torch.sort(flat_tensor, dim=1, descending=descending)  # 按降序排序
        rank_matrix = torch.zeros_like(indices, dtype=torch.float32, device=tensor.device)  # 初始化位次矩阵
        for b in range(B):
            rank_matrix[b, indices[b]] = torch.arange(L * K, dtype=torch.float32, device=tensor.device)  # 生成位次
        rank_matrix = rank_matrix.reshape(B, L, K)  # 恢复为 (B, L, K) 的形状

    else:
        raise ValueError("Input tensor must be 2D (L, K) or 3D (B, L, K).")
    
    return rank_matrix