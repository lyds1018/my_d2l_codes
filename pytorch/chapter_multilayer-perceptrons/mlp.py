import torch
from d2l import torch as d2l
from torch import nn

# 定义模型
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))


# 初始化模型参数
def init_weights(m):
    if type(m) is nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)

# 设置超参数
batch_size, lr, num_epochs = 256, 0.1, 10

# 定义损失函数
loss = nn.CrossEntropyLoss()

# 定义优化器
trainer = torch.optim.SGD(net.parameters(), lr=lr)

# 读取数据集
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


# 训练模型
if __name__ == "__main__":
    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            # 前向计算
            y_hat = net(X)
            l = loss(y_hat, y)  # tensor标量

            # 反向传播和优化
            trainer.zero_grad()
            l.backward()
            trainer.step()

            # 累计损失和准确率
            train_loss_sum += l.item() * y.size(0)  # item()转换为python标量
            train_acc_sum += (
                (y_hat.argmax(dim=1) == y).sum().item()
            )  # 沿着每行取最大值的索引，与真实标签比较
            n += y.size(0)

        # 计算训练集平均损失和准确率
        train_loss = train_loss_sum / n
        train_acc = train_acc_sum / n

        # 测试集准确率
        test_acc_sum, test_n = 0.0, 0
        with torch.no_grad():
            for X_test, y_test in test_iter:
                y_hat_test = net(X_test)
                test_acc_sum += (y_hat_test.argmax(dim=1) == y_test).sum().item()
                test_n += y_test.size(0)
        test_acc = test_acc_sum / test_n

        print(
            f"epoch {epoch + 1}, loss {train_loss:.3f}, "
            f"train acc {train_acc:.3f}, test acc {test_acc:.3f}"
        )
