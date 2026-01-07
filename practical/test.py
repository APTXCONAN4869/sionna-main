# import torch
# print("PyTorch:", torch.__version__)
# print("CUDA available:", torch.cuda.is_available())
# print("Torch CUDA version:", torch.version.cuda)
# print("Device count:", torch.cuda.device_count())
# if torch.cuda.is_available():
#     num_gpus = torch.cuda.device_count()
#     print(f"检测到 {num_gpus} 块 GPU:")
#     for i in range(num_gpus):
#         print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
#     device = torch.device("cuda:3")
# tensor = torch.tensor(1, device=device)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# 加载数据
data = np.load('D:\\sionna-main\\self_loop_data.npy')
buf = np.load('D:\\sionna-main\\buf.npy')
ch1 = np.load('D:\\sionna-main\\ch1.npy')
# 处理数据（同上）
if data.ndim > 1:
    if data.ndim == 2:
        complex_signal = data[:, 0]
    elif data.ndim == 3:
        complex_signal = data[0, 0, :]
    else:
        complex_signal = data.reshape(-1)[:10000]
else:
    complex_signal = data

# 限制点数以便清晰显示
if len(complex_signal) > 1000:
    # 均匀采样
    indices = np.linspace(0, len(complex_signal)-1, 1000, dtype=int)
    complex_signal = complex_signal[indices]

# 创建复平面图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 子图1：散点图（星座图）
scatter = ax1.scatter(np.real(complex_signal), np.imag(complex_signal), 
                      c=np.arange(len(complex_signal)), cmap='viridis',
                      s=10, alpha=0.6, edgecolors='none')
ax1.set_title('constellation Diagram')
ax1.set_xlabel('(I)')
ax1.set_ylabel('(Q)')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)
ax1.set_aspect('equal', 'box')

# 添加单位圆
unit_circle = Circle((0, 0), 1, fill=False, color='red', linestyle='--', alpha=0.5)
ax1.add_patch(unit_circle)

# 子图2：带时间颜色的轨迹图
ax2.scatter(np.real(complex_signal), np.imag(complex_signal), 
           c=np.arange(len(complex_signal)), cmap='rainbow',
           s=15, alpha=0.8, edgecolors='none')
ax2.set_title('Signal Trajectory in IQ Plane')
ax2.set_xlabel('I')
ax2.set_ylabel('Q')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
ax2.set_aspect('equal', 'box')

# 添加箭头显示时间方向
if len(complex_signal) > 10:
    # 在几个关键点添加箭头
    arrow_points = np.linspace(0, len(complex_signal)-2, 5, dtype=int)
    for i in arrow_points:
        dx = np.real(complex_signal[i+1]) - np.real(complex_signal[i])
        dy = np.imag(complex_signal[i+1]) - np.imag(complex_signal[i])
        ax2.arrow(np.real(complex_signal[i]), np.imag(complex_signal[i]), 
                 dx*0.8, dy*0.8, head_width=0.05, head_length=0.1, 
                 fc='black', ec='black', alpha=0.5)

plt.colorbar(scatter, ax=ax1, label='Time Sequence')
plt.tight_layout()
plt.show()

# 分析星座特性
print("Constellation Diagram Analysis:")
print(f"Mean: {np.mean(complex_signal):.4f}{'+' if np.imag(np.mean(complex_signal)) >= 0 else ''}{np.imag(np.mean(complex_signal)):.4f}j")
print(f"Standard Deviation: {np.std(complex_signal):.4f}")
print(f"Power: {np.mean(np.abs(complex_signal)**2):.4f}")