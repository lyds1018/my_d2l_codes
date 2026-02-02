import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from data_set import ImageDataset
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

basedir = os.path.dirname(os.path.abspath(__file__))


# 1. 定义数据增广
train_transform = T.Compose(
    [
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=2, magnitude=4),
        T.ToTensor(),
        T.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ]
)

test_transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ]
)


# 2. 创建数据集
dataset = ImageDataset(
    csv_file=os.path.join(basedir, "data/train.csv"),
    img_path=os.path.join(basedir, "data/train"),
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
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False), nn.BatchNorm2d(out_c)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y += self.shortcut(x)
        return F.relu(y)


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()

        # CIFAR: 3×32×32
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU()
        )

        # 4 stages × 2 blocks
        self.layer1 = nn.Sequential(ResBlock(64, 64), ResBlock(64, 64))

        self.layer2 = nn.Sequential(ResBlock(64, 128, 2), ResBlock(128, 128))

        self.layer3 = nn.Sequential(ResBlock(128, 256, 2), ResBlock(256, 256))

        self.layer4 = nn.Sequential(ResBlock(256, 512, 2), ResBlock(512, 512))

        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)

        return self.fc(x)


# 4. 训练模型
def train_model(model, criterion, optimizer, scheduler, batch_size, num_epochs):
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
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
        scheduler.step()
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
    num_epochs, lr, batch_size, weight_decay = 200, 1e-4, 128, 1e-5
    num_classes = len(dataset.label2idx)

    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], gamma=0.1
    )

    train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        batch_size=batch_size,
        num_epochs=num_epochs,
    )
