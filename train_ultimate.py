import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 读取训练数据
train_df = pd.read_csv('train.csv')

# 提取特征和标签
train_labels = train_df['label'].values
train_features = train_df.drop('label', axis=1).values

# 数据预处理
train_features = train_features / 255.0
train_features = train_features.reshape(-1, 1, 28, 28)

# 创建原始数据和反转数据的组合
train_features_both = np.concatenate([train_features, 1.0 - train_features], axis=0)
train_labels_both = np.concatenate([train_labels, train_labels], axis=0)

# 转换为张量
train_features_tensor = torch.tensor(train_features_both, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels_both, dtype=torch.long)

# 创建数据集和数据加载器
train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# 定义高级CNN模型
class AdvancedCNN(nn.Module):
    def __init__(self):
        super(AdvancedCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# 初始化模型、损失函数和优化器
model = AdvancedCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

# 训练模型
epochs = 30
best_loss = float('inf')

for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    
    # 更新学习率
    scheduler.step(epoch_loss)
    
    # 保存最佳模型
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), 'mnist_cnn_model.pth')
        print('Save new best model')
    
    print('Epoch %d/%d, Loss: %.4f, Accuracy: %.2f%%' % (epoch+1, epochs, epoch_loss, epoch_accuracy))

print('\nTraining complete! Best model saved as mnist_cnn_model.pth')

# 验证模型
model.load_state_dict(torch.load('mnist_cnn_model.pth'))
model.eval()

# 验证原始数据
print('\n--- Validate original data ---')
sample_data = torch.tensor(train_features[:20], dtype=torch.float32)
with torch.no_grad():
    outputs = model(sample_data)
    predictions = torch.argmax(outputs, dim=1).numpy()

correct = 0
for i in range(20):
    label = train_labels[i]
    pred = predictions[i]
    status = 'CORRECT' if label == pred else 'WRONG'
    print('Sample %d: Label=%d, Prediction=%d, %s' % (i+1, label, pred, status))
    if label == pred:
        correct += 1
print('Original data accuracy: %d/20 = %d%%' % (correct, correct*5))

# 验证反转数据
print('\n--- Validate reversed data ---')
reverse_data = torch.tensor(1.0 - train_features[:20], dtype=torch.float32)
with torch.no_grad():
    outputs = model(reverse_data)
    predictions = torch.argmax(outputs, dim=1).numpy()

correct = 0
for i in range(20):
    label = train_labels[i]
    pred = predictions[i]
    status = 'CORRECT' if label == pred else 'WRONG'
    print('Sample %d: Label=%d, Prediction=%d, %s' % (i+1, label, pred, status))
    if label == pred:
        correct += 1
print('Reversed data accuracy: %d/20 = %d%%' % (correct, correct*5))