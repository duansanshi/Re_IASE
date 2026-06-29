import torch
import matplotlib.pyplot as plt
import os

# 加载注意力权重
path = "/home/duanlei/PriSTI/AAAAAAAAAAAAAtention.pt"
attn_weights = torch.load(path).median(dim=1).values
attn_weights = attn_weights.cpu().detach().numpy()
attn_weights = attn_weights

# 确保权重形状是你期望的 (207, 24, 24)
print("Loaded attention weights shape:", attn_weights.shape)

# 创建保存图像的目录
save_dir = "/home/duanlei/PriSTI/attention_heatmaps_aqi_spatial_train"
os.makedirs(save_dir, exist_ok=True)

# 绘制每个热力图并保存
for i in range(attn_weights.shape[0]):
    plt.figure(figsize=(6, 5))  # 设置图像大小
    plt.imshow(attn_weights[i], cmap='viridis', interpolation='nearest')  # 使用viridis颜色图
    plt.colorbar()  # 显示颜色条
    plt.title(f"Attention Map {i+1}")  # 设置标题
    plt.xlabel("Sequence Position")
    plt.ylabel("Sequence Position")
    plt.tight_layout()  # 调整整体布局以防止标签被裁剪

    # 保存图像，文件名为 'heatmap_xxx.png' 格式
    plt.savefig(f"{save_dir}/heatmap_{i+1}.png")
    plt.close()  # 关闭图像以释放内存
