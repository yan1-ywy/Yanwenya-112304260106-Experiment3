import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# 设置随机种子
torch.manual_seed(42)

# 读取训练数据
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 提取特征和标签
train_labels = train_df['label'].values
train_features = train_df.drop('label', axis=1).values
test_features = test_df.values

# 数据预处理
train_features = train_features / 255.0
test_features = test_features / 255.0

# 转换为张量并调整形状
train_features = torch.tensor(train_features, dtype=torch.float32).view(-1, 1, 28, 28)
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_features = torch.tensor(test_features, dtype=torch.float32).view(-1, 1, 28, 28)

# 创建数据集和数据加载器
train_dataset = TensorDataset(train_features, train_labels)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 定义轻量级优化CNN模型
class LightOptimizedCNN(nn.Module):
    def __init__(self):
        super(LightOptimizedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 初始化模型、损失函数和优化器
model = LightOptimizedCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 15
for epoch in range(epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')

# 在测试数据上进行预测
with torch.no_grad():
    test_outputs = model(test_features)
    _, predictions = torch.max(test_outputs, 1)

# 生成submission文件
submission = pd.DataFrame({'ImageId': range(1, len(predictions)+1), 'Label': predictions.numpy()})
submission.to_csv('sample_submission.csv', index=False)

print('Light optimized submission file generated successfully!')