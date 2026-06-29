import matplotlib.pyplot as plt
import numpy as np

models = ['CSDI', 'PriSTI', 'Score-CDM']
methods = ['DDPM', 'FreeCMS', 'OSDI']

columns = ['AQI36', 'PeMS08', 'PeMS04']

data_col1_ddpm = [9.56, 8.90, 11.23]
data_col1_s1   = [9.62, 8.92, 11.33]
data_col1_s2   = [9.45, 9.01, 11.37]

data_col2_ddpm = [10.44, 9.91, 11.13]
data_col2_s1   = [10.45, 9.99, 10.96]
data_col2_s2   = [10.53, 10.10, 11.26]

data_col3_ddpm = [15.44, 14.60, 15.84]
data_col3_s1   = [15.45, 14.58, 15.56]
data_col3_s2   = [15.51, 14.64, 15.59]

all_data = [
    (data_col1_ddpm, data_col1_s1, data_col1_s2),
    (data_col2_ddpm, data_col2_s1, data_col2_s2),
    (data_col3_ddpm, data_col3_s1, data_col3_s2)
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
x = np.arange(len(models))
width = 0.25

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i, ax in enumerate(axes):
    ddpm_data, s1_data, s2_data = all_data[i]
    
    rects1 = ax.bar(x - width, ddpm_data, width, label=methods[0], color=colors[0], edgecolor='black', linewidth=0.5)
    rects2 = ax.bar(x, s1_data, width, label=methods[1], color=colors[1], edgecolor='black', linewidth=0.5)
    rects3 = ax.bar(x + width, s2_data, width, label=methods[2], color=colors[2], edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title(columns[i], fontsize=14, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    
    y_min = min(min(ddpm_data), min(s1_data), min(s2_data))
    y_max = max(max(ddpm_data), max(s1_data), max(s2_data))
    ax.set_ylim(y_min * 0.95, y_max * 1.05) 
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fontsize=12)

plt.tight_layout()
plt.savefig('/home/duanlei/PriSTI/毕业/acceleration_comparison.pdf', format='pdf', bbox_inches='tight')
plt.show()