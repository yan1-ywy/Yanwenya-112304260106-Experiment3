import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import io

# 定义CNN模型
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 3 * 3)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 加载模型
model = CNNModel()
model.load_state_dict(torch.load('mnist_cnn_model.pth', map_location=torch.device('cpu')))
model.eval()

# 读取训练数据
train_df = pd.read_csv('train.csv')

# 测试前10个样本
correct = 0
for i in range(10):
    # 获取原始训练数据
    img_data = train_df.iloc[i, 1:].values.reshape(28, 28)
    label = train_df.iloc[i, 0]
    
    # 归一化
    img_normalized = img_data / 255.0
    
    # 转换为张量
    img_tensor = torch.tensor(img_normalized, dtype=torch.float32).view(1, 1, 28, 28)
    
    # 预测
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.argmax(output, dim=1).item()
    
    print(f"Sample {i+1}: Label={label}, Prediction={prediction}, {'CORRECT' if label == prediction else 'WRONG'}")
    if label == prediction:
        correct += 1

print(f"\n准确率: {correct}/10 = {correct*10}%")

# 测试模拟网页输入（白底黑字转黑底白字）
print("\n--- 测试模拟网页输入 ---")
correct_web = 0
for i in range(10):
    # 获取原始训练数据
    img_data = train_df.iloc[i, 1:].values.reshape(28, 28)
    label = train_df.iloc[i, 0]
    
    # 模拟网页输入（黑底白字）
    img_web = 255 - img_data
    
    # 归一化
    img_normalized = img_web / 255.0
    
    # 转换为张量
    img_tensor = torch.tensor(img_normalized, dtype=torch.float32).view(1, 1, 28, 28)
    
    # 预测
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.argmax(output, dim=1).item()
    
    print(f"Sample {i+1}: Label={label}, Prediction={prediction}, {'CORRECT' if label == prediction else 'WRONG'}")
    if label == prediction:
        correct_web += 1

print(f"\n网页输入模拟准确率: {correct_web}/10 = {correct_web*10}%")