import matplotlib.pyplot as plt
import numpy as np

# 模拟四组实验的Loss数据（基于之前的实验结果）
np.random.seed(42)

# Exp1: SGD, lr=0.01, batch=64, no augmentation, no early stopping
exp1_train_loss = [2.3, 1.2, 0.6, 0.4, 0.32, 0.28, 0.25, 0.22, 0.20, 0.18, 0.16, 0.15, 0.14, 0.13, 0.12]
exp1_val_loss = [2.2, 1.1, 0.55, 0.38, 0.30, 0.26, 0.24, 0.22, 0.21, 0.20, 0.19, 0.18, 0.18, 0.17, 0.17]

# Exp2: Adam, lr=0.001, batch=64, no augmentation, no early stopping
exp2_train_loss = [2.3, 0.8, 0.35, 0.22, 0.16, 0.12, 0.09, 0.07, 0.06, 0.05, 0.04, 0.035, 0.03, 0.027, 0.024]
exp2_val_loss = [2.2, 0.75, 0.32, 0.20, 0.15, 0.12, 0.10, 0.085, 0.075, 0.068, 0.062, 0.058, 0.055, 0.052, 0.05]

# Exp3: Adam, lr=0.001, batch=128, no augmentation, early stopping
exp3_train_loss = [2.3, 0.9, 0.4, 0.25, 0.18, 0.14, 0.11, 0.09, 0.075, 0.065]
exp3_val_loss = [2.2, 0.85, 0.38, 0.22, 0.16, 0.13, 0.11, 0.095, 0.085, 0.08]

# Exp4: Adam, lr=0.001, batch=64, augmentation, early stopping
exp4_train_loss = [2.3, 1.0, 0.5, 0.32, 0.24, 0.19, 0.16, 0.13, 0.11, 0.095, 0.085, 0.077, 0.07, 0.065, 0.06, 0.055, 0.052, 0.05]
exp4_val_loss = [2.2, 0.95, 0.45, 0.28, 0.20, 0.16, 0.14, 0.12, 0.105, 0.095, 0.088, 0.083, 0.079, 0.076, 0.073, 0.071, 0.069, 0.068]

# 创建图像
plt.figure(figsize=(12, 8))

# 绘制训练Loss
plt.plot(range(1, len(exp1_train_loss)+1), exp1_train_loss, 'r-', marker='o', label='Exp1 (SGD) - Train')
plt.plot(range(1, len(exp1_val_loss)+1), exp1_val_loss, 'r--', marker='x', label='Exp1 (SGD) - Val')

plt.plot(range(1, len(exp2_train_loss)+1), exp2_train_loss, 'g-', marker='o', label='Exp2 (Adam) - Train')
plt.plot(range(1, len(exp2_val_loss)+1), exp2_val_loss, 'g--', marker='x', label='Exp2 (Adam) - Val')

plt.plot(range(1, len(exp3_train_loss)+1), exp3_train_loss, 'b-', marker='o', label='Exp3 (Adam+ES) - Train')
plt.plot(range(1, len(exp3_val_loss)+1), exp3_val_loss, 'b--', marker='x', label='Exp3 (Adam+ES) - Val')

plt.plot(range(1, len(exp4_train_loss)+1), exp4_train_loss, 'm-', marker='o', label='Exp4 (Adam+Aug) - Train')
plt.plot(range(1, len(exp4_val_loss)+1), exp4_val_loss, 'm--', marker='x', label='Exp4 (Adam+Aug) - Val')

# 设置图表属性
plt.title('四组对比实验的 Loss 曲线', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(0, 18)
plt.ylim(0, 2.5)

# 保存图片
plt.savefig('loss_curve.png', dpi=150, bbox_inches='tight')
print("Loss曲线图已保存为: loss_curve.png")

# 显示图片
plt.show()