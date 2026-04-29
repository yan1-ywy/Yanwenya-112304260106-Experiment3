from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import base64
from PIL import Image
import io
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 定义高级CNN模型（最优模型结构）
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

# 加载最优模型
model = AdvancedCNN()
model.load_state_dict(torch.load('mnist_cnn_model.pth', map_location=torch.device('cpu')))
model.eval()
print('Best model loaded successfully')

# 图像预处理函数
def preprocess_image(img):
    # 转换为灰度
    img = img.convert('L')
    
    # 转换为numpy数组
    img_np = np.array(img)
    
    # 找到非零像素（手写部分）
    non_zero = np.where(img_np > 20)
    
    if len(non_zero[0]) == 0:
        return np.zeros((28, 28))
    
    # 获取边界
    y_min, y_max = non_zero[0].min(), non_zero[0].max()
    x_min, x_max = non_zero[1].min(), non_zero[1].max()
    
    # 添加边距
    margin = 8
    y_min = max(0, y_min - margin)
    y_max = min(img_np.shape[0] - 1, y_max + margin)
    x_min = max(0, x_min - margin)
    x_max = min(img_np.shape[1] - 1, x_max + margin)
    
    # 裁剪
    cropped = img_np[y_min:y_max+1, x_min:x_max+1]
    
    # 计算目标大小
    height, width = cropped.shape
    max_dim = max(height, width)
    
    # 创建28x28的画布
    final_img = np.zeros((28, 28), dtype=np.float32)
    
    # 计算缩放比例 - 让数字占据约80%的图像
    scale = (28 * 0.8) / max_dim
    new_height = int(height * scale)
    new_width = int(width * scale)
    
    # 确保至少有1像素
    new_height = max(1, new_height)
    new_width = max(1, new_width)
    
    # 使用PIL调整大小
    cropped_pil = Image.fromarray(cropped)
    resized = cropped_pil.resize((new_width, new_height), Image.LANCZOS)
    resized_np = np.array(resized)
    
    # 计算居中位置
    y_offset = (28 - new_height) // 2
    x_offset = (28 - new_width) // 2
    
    # 放置到中心
    final_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_np
    
    # 归一化到0-1
    final_img = final_img / 255.0
    
    return final_img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)
    
    # 处理图像
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    processed_img = preprocess_image(img)
    
    # 添加批次维度和通道维度
    img_tensor = torch.tensor(processed_img, dtype=torch.float32).view(1, 1, 28, 28)
    
    # 预测
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][prediction].item() * 100
    
    return jsonify({
        'prediction': prediction,
        'confidence': round(confidence, 2),
        'probabilities': probabilities[0].tolist()
    })

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file:
        # 处理上传的图像
        img = Image.open(file.stream).convert('L')
        processed_img = preprocess_image(img)
        
        # 添加批次维度和通道维度
        img_tensor = torch.tensor(processed_img, dtype=torch.float32).view(1, 1, 28, 28)
        
        # 预测
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            prediction = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][prediction].item() * 100
        
        return jsonify({
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'probabilities': probabilities[0].tolist()
        })

if __name__ == '__main__':
    app.run(debug=True)