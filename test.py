import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# --- 关键配置：确保字体合规 ---
matplotlib.rcParams['pdf.fonttype'] = 42  # 强制使用 TrueType 字体 (Type 42)，避免 Type 3
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = False # 如果环境没装完整 LaTeX，先设为 False
# 强制使用标准衬线字体，这通常能更好地映射 Unicode
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman'] + matplotlib.rcParams['font.serif']
# 1. 构造数据
# data = [
#     ["PriSTI", "aqi", 0.30, 9.02], ["PriSTI", "aqi", 0.45, 9.04],
#     ["PriSTI", "aqi", 0.60, 9.01], ["PriSTI", "aqi", 0.75, 9.06],
#     ["PriSTI", "aqi", 0.90, 9.03],
#     ["PriSTI", "pems04", 0.30, 14.65], ["PriSTI", "pems04", 0.45, 14.67],
#     ["PriSTI", "pems04", 0.60, 14.64], ["PriSTI", "pems04", 0.75, 14.67],
#     ["PriSTI", "pems04", 0.90, 14.67],
#     ["PriSTI", "pems08", 0.30, 10.12], ["PriSTI", "pems08", 0.45, 10.11],
#     ["PriSTI", "pems08", 0.60, 10.17], ["PriSTI", "pems08", 0.75, 10.14],
#     ["PriSTI", "pems08", 0.90, 10.10],
# ]
data = [
    ["CSDI", "aqi", 0.30, 9.44], ["CSDI", "aqi", 0.45, 9.46],
    ["CSDI", "aqi", 0.60, 9.45], ["CSDI", "aqi", 0.75, 9.46],
    ["CSDI", "aqi", 0.90, 9.42],
    ["CSDI", "pems04", 0.30, 15.51], ["CSDI", "pems04", 0.45, 15.51],
    ["CSDI", "pems04", 0.60, 15.51], ["CSDI", "pems04", 0.75, 15.54],
    ["CSDI", "pems04", 0.90, 15.53],
    ["CSDI", "pems08", 0.30, 10.53], ["CSDI", "pems08", 0.45, 10.53],
    ["CSDI", "pems08", 0.60, 10.53], ["CSDI", "pems08", 0.75, 10.53],
    ["CSDI", "pems08", 0.90, 10.54],
]
# data = [
#     ["Score-CDM", "aqi", 0.30, 11.42], ["Score-CDM", "aqi", 0.45, 11.52],
#     ["Score-CDM", "aqi", 0.60, 11.47], ["Score-CDM", "aqi", 0.75, 11.41],
#     ["Score-CDM", "aqi", 0.90, 11.37],
#     ["Score-CDM", "pems04", 0.30, 15.67], ["Score-CDM", "pems04", 0.45, 15.66],
#     ["Score-CDM", "pems04", 0.60, 15.71], ["Score-CDM", "pems04", 0.75, 15.59],
#     ["Score-CDM", "pems04", 0.90, 15.71],
#     ["Score-CDM", "pems08", 0.30, 11.28], ["Score-CDM", "pems08", 0.45, 11.27],
#     ["Score-CDM", "pems08", 0.60, 11.26], ["Score-CDM", "pems08", 0.75, 11.29],
#     ["Score-CDM", "pems08", 0.90, 11.28],
# ]

df = pd.DataFrame(data, columns=["Model", "Dataset", "q", "MAE"])
df.to_csv("quantile_experiment_results.csv", index=False)

# 2. 设置绘图风格
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
datasets = ["aqi", "pems04", "pems08"]
colors = ["#4C72B0", "#55A868", "#C44E52"]

# 3. 逐个数据集绘图
for i, ds in enumerate(datasets):
    subset = df[df["Dataset"] == ds]
    axes[i].plot(subset["q"], subset["MAE"], marker='o', linestyle='-', 
                 color=colors[i], linewidth=2, markersize=8)
    axes[i].set_title(f"Dataset: {ds.upper()}", fontsize=14, fontweight='bold')
    axes[i].set_xlabel(r"Quantile $q$", fontsize=12)
    axes[i].set_ylabel("MAE", fontsize=12)
    
    # 动态调整坐标轴以显示微小差异
    y_min, y_max = subset["MAE"].min(), subset["MAE"].max()
    margin = (y_max - y_min) * 0.5 if y_max != y_min else 0.05
    axes[i].set_ylim(y_min - margin, y_max + margin)

plt.tight_layout()
plt.savefig("quantile_q_sensitivity.pdf", dpi=300)