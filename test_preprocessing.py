import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取训练数据
train_df = pd.read_csv('train.csv')

# 获取第一张图片
img_data = train_df.iloc[0, 1:].values.reshape(28, 28)
label = train_df.iloc[0, 0]

print(f"训练数据 - Label: {label}")
print(f"训练数据 - Min: {img_data.min()}, Max: {img_data.max()}")
print(f"训练数据 - 数据类型: {img_data.dtype}")

# 显示原始训练图像
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.imshow(img_data, cmap='gray')
plt.title(f'MNIST训练数据 (Label: {label})')
plt.axis('off')

# 模拟网页端的预处理流程
# 网页画布是黑底白字，所以需要反转
img_reversed = 255 - img_data
plt.subplot(1, 2, 2)
plt.imshow(img_reversed, cmap='gray')
plt.title('反转后（模拟网页输入）')
plt.axis('off')

plt.show()

# 检查像素分布
print("\n训练数据像素统计:")
print(f"黑色像素（0）数量: {np.sum(img_data == 0)}")
print(f"白色像素（255）数量: {np.sum(img_data == 255)}")
print(f"非零像素比例: {np.sum(img_data > 0) / (28*28):.2%}")