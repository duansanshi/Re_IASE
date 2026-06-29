# Codex Notes for PriSTI / NewImputer / UNET

本文档记录 `/home/duanlei/PriSTI` 中 IASE-Net 相关代码、论文设定和实验参数，供后续 Codex 接手时直接使用。

## 之前对话总结

之前主要完成了以下整理工作：

- 梳理了论文 `cotent.tex` 中 IASE-Net 的核心设定：预插值输入增强、U-Net 风格层次化时空网络、自适应中间监督、可选时间平滑损失。
- 明确了 IASE-Net 相关代码主要位于 `NewImputer/UNET`，并整理了 AQI36、PEMS04、PEMS08 三条实验主线对应的 trainer、datamodule、dataset、model、LightningModule 文件关系。
- 记录了 AQI36 当前主线的关键行为：输入为 `[coeffs, cond_mask]`，`coeffs` 来自预插补和可选频域低通，不是简单的 `observed_data * cond_mask`。
- 整理了训练损失形式：最终层 masked L1、中间层自适应渐进监督、时间差分平滑损失，以及 validation/test 的指标计算位置。
- 汇总了论文表格中的主要超参数、主表 MAE、消融结果和 AQI36 可视化结果，方便后续复现实验或核对论文描述。
- 标注了容易出错的版本关系，例如 `v5`、`v6`、`v8`、`v10` 的输入字段和损失项差异，以及运行 `v6` 频率辅助输入前必须确认 batch 是否真的包含 `freq_coeffs`。
- 记录了常用训练、测试、批量 retest 命令模板，并提醒后续改动前优先确认数据集、入口 trainer 和 batch 字段一致性。
- 当前新要求是：为了避免后续 Codex 修改影响原有代码，先通过 Git 做版本管理和保护性快照，再继续进行代码修改。



## 环境

- 推荐环境：`mambaimputer`
- Conda 初始化方式：

```bash
source /home/duanlei/anaconda3/etc/profile.d/conda.sh && conda activate mambaimputer
```

- 已验证 Python：

```text
Python 3.10.13
/home/duanlei/anaconda3/envs/mambaimputer/bin/python
```

- 论文中记录的软件环境：
  - Python 3.10.13
  - PyTorch 2.1.1+cu118
  - PyTorch Lightning 2.5.1
  - Pandas 2.2.3
  - NumPy 1.26.4

## 论文对应关系

论文文件：`/home/duanlei/PriSTI/cotent.tex`

章节主题是 IASE-Net：输入与自适应监督增强的层次化时空插补网络（Input and Adaptive Supervision Enhancement Network）。

核心思想：

- 非扩散单步插补模型，目标是低时延、轻量化部署。
- 输入增强：先用简单插值构造 `X^s`，再与掩码 `M` 拼接输入模型。
- 层次化时空网络：U-Net 风格上下采样 + 时间/变量维 Transformer。
- 自适应监督增强：中间层预测接受软权重或硬分位数渐进监督，最终层全局监督。
- 可选时间平滑损失：约束预测序列差分接近真实序列差分。

## 主要目录

IASE-Net 相关代码集中在：

```text
/home/duanlei/PriSTI/NewImputer/UNET
```

该目录包含源码、训练脚本、日志、checkpoint、可视化图片和论文片段。阅读/修改时优先关注 `.py` 和 `.sh` 文件，通常跳过：

- `logs/`
- `checkpoints/`
- `pictures/`
- `__pycache__/`

## AQI36 主线

当前 AQI36 主线大致为：

```text
aqi36_lightning_trainer_new_v5.py
  -> aqi36_lightning_datamodule.py
  -> dataset_aqi.py
  -> transformer_imputer_new.py
  -> imputer_lightning_module_new_v5.py
```

关键文件：

- `aqi36_lightning_trainer_new_v5.py`
  - 训练入口。
  - 默认 `epochs=200`，`lr=5e-4`。
  - 默认 `preimpute=Forward`。
  - 默认启用频域低通输入：`freq_flag=not args.no_freq`。
  - 默认 `curriculum_weight=0.3`，`curriculum_warmup_ratio=0.1`，`curriculum_temperature=1.0`，`smooth_weight=0.03`。

- `aqi36_lightning_datamodule.py`
  - LightningDataModule。
  - 默认 `is_interpolate=True`。
  - 将参数传给 `dataset_aqi.get_dataloader`。

- `dataset_aqi.py`
  - 读取 `/home/duanlei/PriSTI/data/pm25/SampleData/pm25_ground.txt` 和 `pm25_missing.txt`。
  - 使用 `/home/duanlei/PriSTI/data/pm25/pm25_meanstd.pk` 标准化。
  - 输出 batch 字段包括 `observed_data`、`observed_mask`、`gt_mask`、`hist_mask`、`cond_mask`、`coeffs` 等。
  - `coeffs` 是预插补后可选频域低通滤波的输入，不是简单的 `observed_data * cond_mask`。

- `transformer_imputer_new.py`
  - IASE-Net 主体模型。
  - 输入为 `[coeffs, cond_mask]` 两通道。
  - 时间位置编码为 sinusoidal，特征位置编码为可学习参数。
  - 堆叠 3 个 `UnetBlock`。
  - 每个 block 后通过共享 `Linear-ReLU-Linear` 输出一个 `layer_pred`。
  - 当前最终输出是 `skip_results[-1]`，保留了 dynamic gating 代码但未启用。

- `imputer_lightning_module_new_v5.py`
  - 训练损失：

```text
loss = L_main + alpha * L_curriculum + smooth_weight * smooth_loss
```

  - `L_main`：最终层在 `eval_mask = observed_mask - cond_mask` 上的 masked L1。
  - `L_curriculum`：中间层软难度渐进监督。
  - `smooth_loss`：时间差分平滑约束。
  - validation 使用 masked MSE。
  - test 输出反标准化 MAE/MSE。

## PEMS04 / PEMS08 主线

PEMS04：

```text
pems04_lightning_trainer_v5.py
  -> pems04_lightning_datamodule.py
  -> dataset_pems04.py
  -> transformer_imputer_pems04.py
  -> pems04_lightning_module_v5.py
```

PEMS08：

```text
pems08_lightning_trainer_v5.py
  -> pems08_lightning_datamodule.py
  -> dataset_pems08.py
  -> transfomer_imputer_pems08.py
  -> pems08_lightning_module_v5.py
```

PEMS 数据管线特点：

- 数据来自：
  - `./data/PEMS04/pems04.h5`
  - `./data/PEMS08/pems08.h5`
- 标准化文件：
  - `./data/PEMS04/pems04_meanstd.pk`
  - `./data/PEMS08/pems08_meanstd.pk`
- `eval_length=24`。
- train/valid/test 按 70%/10%/20% 切分。
- 默认 `preimpute_flag="Backward"`。
- 默认 `freq_flag=True`，对预插补结果做 FFT 低通滤波后作为 `coeffs`。
- PEMS04 特征维度 `K=307`。
- PEMS08 特征维度 `K=170`。

## 模型结构参数

论文核心超参数：

| 参数 | AQI36 | PeMS04 | PeMS08 |
| --- | --- | --- | --- |
| 输入构成 | 预插值 + 掩码 | 预插值 + 掩码 | 预插值 + 掩码 |
| 预插值形式 | 前向插值 | 后向插值 | 后向插值 |
| 序列长度 | 36 | 24 | 24 |
| 特征维度 | 36 | 307 | 170 |
| `d_model` | 64 | 64 | 64 |
| 时间位置编码维度 | 128 | 128 | 128 |
| 特征位置编码维度 | 16 | 16 | 16 |
| 模块堆叠数 | 3 | 3 | 3 |
| 下采样比例 | L -> L/3 -> L/12 | L -> L/3 -> L/6 | L -> L/3 -> L/6 |
| Transformer heads | 8 | 8 | 8 |
| Dropout | 0.1 | 0.1 | 0.1 |
| 每个 Transformer 子模块层数 | 1 | 1 | 1 |
| 渐进监督权重 | 0.3 | 0.3 | 0.3 |
| 热身比例 | 0.1 | 0.1 | 0.1 |
| 温度系数 | 0.5 | - | 0.5 |
| 时间平滑权重 | 0.03 | 0 | 0.01 |
| 训练轮数 | 200 | 40 | 40 |
| 学习率 | 5e-4 | 5e-4 | 5e-4 |

注意：论文表中 PeMS04 写的是硬分位数划分，当前读到的 `pems04_lightning_module_v5.py` 实现是软权重方式；若需要完全复现实验表，应再核对 `v4`、cascade 或其他 trainer/module 版本。

## 实验结果

论文主表 MAE：

| 方法 | AQI36 | PeMS04 Point | PeMS08 Point |
| --- | ---: | ---: | ---: |
| PriSTI | 8.90 | 14.60 | 9.91 |
| CSDI | 9.56 | 15.44 | 10.44 |
| Score-CDM | 11.23 | 15.84 | 11.13 |
| ImputeFormer | 11.55 | 17.25 | 12.91 |
| IASE-Net | 8.58 | 14.18 | 9.59 |

消融表：

| 设置 | AQI36 | PeMS04 | PeMS08 |
| --- | ---: | ---: | ---: |
| IASE-Net | 8.58 | 14.18 | 9.59 |
| 线性插值预输入 | 9.44 | 15.08 | 10.32 |
| 移除预插值 | 9.49 | 15.48 | 10.58 |
| 移除自适应监督，仅全局 L1 | 8.82 | 14.25 | 9.70 |

AQI36 可视化结果：

- sensor 20：26 个缺失点，MAE 8.82。
- sensor 33：16 个缺失点，MAE 7.99。

## 版本关系和注意事项

AQI 相关版本：

- `transformer_imputer_new.py` + `imputer_lightning_module_new_v5.py`
  - 当前基础主线。
  - 输入为两通道 `[coeffs, cond_mask]`。
  - 软自适应监督 + 时间平滑。

- `transformer_imputer_v6.py` + `imputer_lightning_module_v6.py`
  - 频率辅助输入版本。
  - 模型输入为三通道 `[raw_preimputed, mask, freq_filtered]`。
  - 依赖 batch 中存在 `freq_coeffs`。

- `imputer_lightning_module_v8.py`
  - 增加频域谱损失 `freq_loss`。
  - 不改变模型输入。

- `transformer_imputer_v10.py`
  - 加入 gap-length embedding，按连续缺失段长度给模型提示。

重要坑点：

- 当前 `aqi36_lightning_datamodule.py` 导入的是 `dataset_aqi.py`。
- 当前 `dataset_aqi.py` 接收了一些参数，如 `mavg_window`、`decay_alpha`、`dct_ratio`、`wavelet_level`、`aux_freq`，但实际主逻辑并没有完整使用这些参数。
- `dataset_aqi_backup_v6.py` 比当前 `dataset_aqi.py` 功能更全，包含 `aux_freq`、`freq_coeffs`、移动平均、DCT、小波等逻辑。
- 如果要跑 `v6` 频率辅助输入，必须确认数据集实际能产出 `freq_coeffs`。仅 trainer 传 `--aux_freq` 不一定有效。
- 多数 trainer 会把 checkpoint 写到 `checkpoints/myimputer_*`，日志写到 `logs/unet_imputer_*`。
- 训练和测试指标只在 `eval_mask = observed_mask - cond_mask` 指定的位置统计。

## 常用命令模板

AQI36 v5 训练并测试：

```bash
source /home/duanlei/anaconda3/etc/profile.d/conda.sh && conda activate mambaimputer && \
cd /home/duanlei/PriSTI/NewImputer/UNET && \
python aqi36_lightning_trainer_new_v5.py --device 0 --epochs 200 --lr 5e-4 \
  --preimpute Forward --curriculum_weight 0.3 --curriculum_warmup_ratio 0.1 \
  --curriculum_temperature 0.5 --smooth_weight 0.03
```

AQI36 只测试 checkpoint：

```bash
source /home/duanlei/anaconda3/etc/profile.d/conda.sh && conda activate mambaimputer && \
cd /home/duanlei/PriSTI/NewImputer/UNET && \
python aqi36_lightning_trainer_new_v5.py --device 0 --test_only \
  --checkpoint_path checkpoints/myimputer_aqi36_EXPNAME/best_model.ckpt
```

批量 retest 脚本：

```bash
source /home/duanlei/anaconda3/etc/profile.d/conda.sh && conda activate mambaimputer && \
cd /home/duanlei/PriSTI/NewImputer/UNET && \
bash run_batch_retest.sh
```

## 修改建议

后续改动前优先确认三件事：

1. 要改的是 AQI36、PEMS04 还是 PEMS08。
2. 入口 trainer 对应的是哪个 model/module/dataset 版本。
3. batch 中是否真的包含模型 forward 需要的字段，尤其是 `coeffs` 和 `freq_coeffs`。

如果新增输入通道或损失项，需要同步检查：

- dataset 是否产出对应字段。
- model `process_data` 是否取出并 permute 到 `[B, K, L]`。
- LightningModule 的 `training_step`、`validation_step`、`test_step` 是否使用一致。
- checkpoint 加载是否会因模型参数形状变化而失效。

## AGENTS.md 说明

项目根目录 `AGENTS.md` 要求完成工作或需要澄清时调用 `askQuestions` 工具。但当前 Codex 会话没有提供该工具，因此无法执行该要求；如工具可用，应在结束前调用它获取下一步指示。
