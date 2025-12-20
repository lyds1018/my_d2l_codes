import os

import torch
import torch.nn as nn
import timm
import torchvision.transforms as transforms
from data_set import ImageDataset
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

basedir = os.path.dirname(os.path.abspath(__file__))

# 1. 定义数据增强
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# 2. 创建数据集
dataset = ImageDataset(
    csv_file=os.path.join(basedir, "data/train.csv"),
    img_path=os.path.join(basedir, "data"),
)
test_ratio = 0.2
test_size = int(len(dataset) * test_ratio)
train_size = len(dataset) - test_size
generator = torch.Generator().manual_seed(42)
train_dataset, test_dataset = random_split(
    dataset, [train_size, test_size], generator=generator
)
train_dataset.dataset.transform = train_transform
test_dataset.dataset.transform = test_transform


# 3. 定义卷积神经网络
class MyCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = timm.create_model("resnext50_32x4d", pretrained=True)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.resnet(x)


# 4. 训练模型
def train_model(model, criterion, optimizer, scheduler, batch_size, num_epochs):
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
    )

    # 训练模型
    device = next(model.parameters()).device  # 获取模型所在设备
    alpha = 0.8  # 测试损失权重
    best_score = float("inf")
    best_epoch = -1

    print("开始训练模型...")
    for epoch in range(num_epochs):
        # 设置训练模式
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"
        ):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        running_loss = running_loss / len(train_loader.dataset)

        # 在测试集上评估模型
        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            for images, labels in tqdm(test_loader, desc="Evaluating"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
        
        test_loss = test_loss / len(test_loader.dataset)
        scheduler.step(test_loss)

        print(f"Train Loss: {running_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f}")

        # 保存目前最优模型
        current_score = alpha * test_loss + (1 - alpha) * running_loss
        if current_score < best_score:
            best_score = current_score
            best_epoch = epoch + 1
            torch.save(
                model.state_dict(), os.path.join(basedir, "model/best_model.pth")
            )

    print(f"最优模型，Epoch: {best_epoch}, Test Loss: {best_score:.4f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs, lr, batch_size, weight_decay = 50, 1e-4, 128, 1e-5
    num_classes = len(dataset.label2idx)

    model = MyCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.3,     
        patience=4,      
        threshold=1e-3
    )


    train_model(
        model, criterion, optimizer, scheduler, batch_size=batch_size, num_epochs=num_epochs
    )