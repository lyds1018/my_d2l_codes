import os

import pandas as pd
import torch
from data_set import ImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from train import MyCNN

# 1. 配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
basedir = os.path.dirname(os.path.abspath(__file__))
os.chdir(basedir)

# 2. 定义数据增强
test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# 3. 创建数据集和数据加载器
train_dataset = ImageDataset(csv_file="data/train.csv", img_path="data")
test_dataset = ImageDataset(
    csv_file="data/test.csv", img_path="data", transform=test_transform, indexed=True
)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 4. 加载模型
num_classes = len(train_dataset.label2idx)
model = MyCNN(num_classes=num_classes)
model.load_state_dict(torch.load("./model/best_model.pth", map_location=device))
model.to(device)
model.eval()

# 5. 预测
preds = []

with torch.no_grad():
    for images in test_loader:
        images = images.to(device)
        outputs = model(images)
        predicted = torch.argmax(outputs, dim=1).cpu().numpy()  # 获取预测类别索引
        preds.extend(predicted)

# 6. 生成提交文件
idx2label = {v: k for k, v in train_dataset.label2idx.items()}
pred_labels = [idx2label[i] for i in preds]

test_df = pd.read_csv("data/test.csv")
submission_df = pd.DataFrame(
    {test_df.columns[0]: test_df.iloc[:, 0], "label": pred_labels}
)

submission_df.to_csv("./result/submission.csv", index=False)
print("预测结果已保存到 ./result/submission.csv")
