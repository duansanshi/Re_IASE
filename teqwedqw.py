# import numpy as np
# import matplotlib.pyplot as plt

# # ==========================================
# # 1. 生成更平缓的模拟时间序列数据
# # ==========================================
# np.random.seed(42)
# L = 300
# t = np.linspace(0, 4*np.pi, L)

# # 减小了高频成分的频率(如10t变5t)和振幅
# y1 = np.sin(t) + 0.15 * np.sin(5*t) + 0.1 * np.random.randn(L)
# y2 = 0.8 * np.cos(t - 0.5) + 0.1 * np.sin(6*t) + 0.1 * np.random.randn(L) + 1.5
# y3 = 1.2 * np.sin(0.8*t + 1) + 0.2 * np.sin(7*t) + 0.1 * np.random.randn(L) - 1.5

# X_true = np.vstack([y1, y2, y3]).T 

# # ==========================================
# # 2. 生成缺失掩码并制造缺失数据
# # ==========================================
# M = np.ones((L, 3), dtype=bool)
# M[50:90, :] = False
# M[140:190, :] = False
# M[240:280, :] = False

# X_masked = np.where(M, X_true, np.nan)

# # ==========================================
# # 3. 绘制纯净组件 (a) - 原始序列
# # ==========================================
# fig, ax = plt.subplots(figsize=(6, 2)) # 调整合适的宽长比
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# for i in range(3):
#     # 画出真实的灰色虚线作为背景参考
#     ax.plot(t, X_true[:, i], color='gray', linestyle='--', alpha=0.5)
#     # 画出观测到的数据
#     ax.plot(t, X_masked[:, i], color=colors[i], linewidth=2)

# # 移除所有坐标轴、刻度和边框
# ax.axis('off') 

# # ==========================================
# # 4. 保存为 SVG (紧凑边界，无多余留白)
# # ==========================================
# plt.savefig('component_a_timeseries.svg', format='svg', bbox_inches='tight', pad_inches=0, transparent=True)
# plt.close()

# import numpy as np
# import matplotlib.pyplot as plt

# # ==========================================
# # 1. 保持与时间序列一致的长度和缺失区间
# # ==========================================
# L = 300

# # 生成掩码，1表示观测到（黄色），0表示缺失（深紫色）
# M = np.ones(L, dtype=int)
# M[50:90] = 0
# M[140:190] = 0
# M[240:280] = 0

# # ==========================================
# # 2. 绘制纯净组件 - 掩码色块
# # ==========================================
# # 设置为扁长的条带形状，宽度与上一个图保持比例一致
# fig, ax = plt.subplots(figsize=(6, 0.8)) 

# # 使用 imshow 绘制 1D 数组，cmap='viridis' 会将 1 映射为黄色，0 映射为深紫色
# ax.imshow(M[np.newaxis, :], aspect='auto', cmap='viridis', extent=[0, L, 0, 1])

# # 移除所有坐标轴、边框、刻度
# ax.axis('off')

# # ==========================================
# # 3. 保存为 SVG (紧凑边界，无多余留白，透明背景)
# # ==========================================
# plt.savefig('component_a_mask.svg', format='svg', bbox_inches='tight', pad_inches=0, transparent=True)
# plt.close()


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # ==========================================
# # 1. 生成完全一致的模拟数据
# # ==========================================
# np.random.seed(42)
# L = 300
# t = np.linspace(0, 4*np.pi, L)

# y1 = np.sin(t) + 0.15 * np.sin(5*t) + 0.1 * np.random.randn(L)
# y2 = 0.8 * np.cos(t - 0.5) + 0.1 * np.sin(6*t) + 0.1 * np.random.randn(L) + 1.5
# y3 = 1.2 * np.sin(0.8*t + 1) + 0.2 * np.sin(7*t) + 0.1 * np.random.randn(L) - 1.5

# X_true = np.vstack([y1, y2, y3]).T 

# M = np.ones((L, 3), dtype=bool)
# M[50:90, :] = False
# M[140:190, :] = False
# M[240:280, :] = False

# X_masked = np.where(M, X_true, np.nan)

# # ==========================================
# # 2. 修改为：前向填充 (Forward Fill / LOCF)
# # ==========================================
# X_df = pd.DataFrame(X_masked)
# # 使用 ffill (forward fill)，用前一个有效值向后填补
# # 加上 bfill() 是为了兜底，防止序列最开头就是 NaN
# X_ff = X_df.ffill().bfill().values 

# # ==========================================
# # 3. 绘制纯净组件 (b) - 前向填充序列
# # ==========================================
# fig, ax = plt.subplots(figsize=(6, 2)) 
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# for i in range(3):
#     ax.plot(t, X_ff[:, i], color=colors[i], linewidth=2)

# ax.axis('off') 

# # ==========================================
# # 4. 保存为 SVG
# # ==========================================
# plt.savefig('component_b_forward_fill.svg', format='svg', bbox_inches='tight', pad_inches=0, transparent=True)
# plt.close()

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # ==========================================
# # 1. 生成完全一致的模拟数据
# # ==========================================
# np.random.seed(42)
# L = 300
# t = np.linspace(0, 4*np.pi, L)

# y1 = np.sin(t) + 0.15 * np.sin(5*t) + 0.1 * np.random.randn(L)
# y2 = 0.8 * np.cos(t - 0.5) + 0.1 * np.sin(6*t) + 0.1 * np.random.randn(L) + 1.5
# y3 = 1.2 * np.sin(0.8*t + 1) + 0.2 * np.sin(7*t) + 0.1 * np.random.randn(L) - 1.5

# X_true = np.vstack([y1, y2, y3]).T 

# M = np.ones((L, 3), dtype=bool)
# M[50:90, :] = False
# M[140:190, :] = False
# M[240:280, :] = False

# X_masked = np.where(M, X_true, np.nan)

# # 前向填充 (Forward Fill)
# X_df = pd.DataFrame(X_masked)
# X_ff = X_df.ffill().bfill().values 

# # ==========================================
# # 2. 计算频域数据 (以第一个变量为例)
# # ==========================================
# # 进行快速傅里叶变换并取绝对值(振幅)
# fft_visual = np.abs(np.fft.fft(X_ff[:, 0]))
# freqs_visual = np.fft.fftfreq(L)

# # 只取正频率部分用于可视化展示
# pos_mask = freqs_visual > 0
# freqs_pos = freqs_visual[pos_mask]
# fft_pos = fft_visual[pos_mask]

# # ==========================================
# # 3. 绘制纯净组件 (c) - 频谱图与截断区域
# # ==========================================
# fig, ax = plt.subplots(figsize=(4, 2)) # 频谱图稍微短一点，符合原图排版比例

# # 绘制蓝色频谱曲线
# ax.plot(freqs_pos, fft_pos, color='#1f77b4', linewidth=1.5)

# # 计算截断位置 (保留前80%，截断后20%)
# max_freq = freqs_pos.max()
# cutoff_freq = max_freq * 0.8

# # 添加红色高频截断阴影区域 (带网格线剖面效果)
# ax.axvspan(cutoff_freq, max_freq, facecolor='lightcoral', alpha=0.5, edgecolor='red', hatch='xx')

# # 使用绘图标记(marker)画一个纯图形的大红叉，不使用任何文本字符
# center_x = cutoff_freq + (max_freq - cutoff_freq) / 2
# center_y = max(fft_pos) * 0.4 # 放在稍微靠中间偏下的位置
# ax.plot(center_x, center_y, marker='x', markersize=35, color='red', markeredgewidth=4)

# # 移除所有坐标轴、边框、刻度
# ax.axis('off') 

# # ==========================================
# # 4. 保存为 SVG
# # ==========================================
# plt.savefig('component_c_spectrum.svg', format='svg', bbox_inches='tight', pad_inches=0, transparent=True)
# plt.close()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. 生成完全一致的模拟数据
# ==========================================
np.random.seed(42)
L = 300
t = np.linspace(0, 4*np.pi, L)

y1 = np.sin(t) + 0.15 * np.sin(5*t) + 0.1 * np.random.randn(L)
y2 = 0.8 * np.cos(t - 0.5) + 0.1 * np.sin(6*t) + 0.1 * np.random.randn(L) + 1.5
y3 = 1.2 * np.sin(0.8*t + 1) + 0.2 * np.sin(7*t) + 0.1 * np.random.randn(L) - 1.5

X_true = np.vstack([y1, y2, y3]).T 

M = np.ones((L, 3), dtype=bool)
M[50:90, :] = False
M[140:190, :] = False
M[240:280, :] = False

X_masked = np.where(M, X_true, np.nan)

# ==========================================
# 2. 前向填充 (Forward Fill)
# ==========================================
X_df = pd.DataFrame(X_masked)
X_ff = X_df.ffill().bfill().values 

# ==========================================
# 3. 核心算法 - 频域高频截断与逆变换
# ==========================================
X_lp = np.zeros_like(X_ff)

for i in range(3):
    # 进行傅里叶变换
    fft_vals = np.fft.fft(X_ff[:, i])
    
    # 计算需要保留的频率范围 (保留低频的 80%)
    keep_ratio = 0.8
    keep_N = int(L * keep_ratio / 2)
    
    # 截断高频部分 (将其设为 0)
    # 频率数组结构: [0, 正低频..., 正高频..., 负高频..., 负低频...]
    fft_vals_filtered = fft_vals.copy()
    fft_vals_filtered[keep_N : -keep_N] = 0 
    
    # 逆傅里叶变换还原为时域信号
    X_lp[:, i] = np.fft.ifft(fft_vals_filtered).real

# ==========================================
# 4. 绘制纯净组件 (c) - 平滑处理后的最终序列
# ==========================================
fig, ax = plt.subplots(figsize=(6, 2)) # 保持与前面序列图一致的尺寸
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i in range(3):
    # 画出最终经过低通滤波平滑后的数据
    ax.plot(t, X_lp[:, i], color=colors[i], linewidth=2)

# 移除所有坐标轴、边框、刻度
ax.axis('off') 

# ==========================================
# 5. 保存为 SVG
# ==========================================
plt.savefig('component_c_smoothed.svg', format='svg', bbox_inches='tight', pad_inches=0, transparent=True)
plt.close()