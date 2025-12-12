import os

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

basedir = os.path.dirname(os.path.abspath(__file__))
os.chdir(basedir)

# --- 加载处理好的数据 ---
train_features = torch.load("./processed_data/train_features.pt")
train_labels = torch.load("./processed_data/train_labels.pt")


# --- 定义模型、评价指标和优化器 ---
loss = nn.MSELoss()
in_features = train_features.shape[1]

# 模型
def get_net():
    net = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 1),
    )
    return net

# 评价指标：对数均方根误差
def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    preds = net(features).clamp(min=1)

    # 计算 log(y) 和 log(y_hat) 之间的 MSE 的平方根
    rmse = torch.sqrt(loss(torch.log(preds), torch.log(labels)))
    return rmse.item()

# 优化器与训练过程
def train(
    net,
    train_features,
    train_labels,
    test_features,
    test_labels,
    num_epochs,
    learning_rate,
    weight_decay,
    batch_size,
):
    # 创建数据迭代器
    train_dataset = TensorDataset(train_features, train_labels)
    train_iter = DataLoader(train_dataset, batch_size, shuffle=True)

    # 使用Adam优化器，weight_decay实现L2正则化
    optimizer = torch.optim.Adam(
        net.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # 训练过程
    train_ls, test_ls = [], []
    for epoch in range(num_epochs):
        net.train()  # 设置为训练模式
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()

        net.eval()  # 设置为评估模式
        with torch.no_grad():
            # 计算训练集上的 log_rmse
            train_ls.append(log_rmse(net, train_features, train_labels))
            # 计算验证/测试集上的 log_rmse
            if test_labels is not None:
                test_ls.append(log_rmse(net, test_features, test_labels))

    return train_ls, test_ls


# --- 超参数调优（k折交叉验证）---
# 获取第i折交叉验证所需要的训练和验证数据
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k

    # 使用 PyTorch 切片和 torch.cat 来构建训练集和验证集
    X_train, y_train = [], []

    for j in range(k):
        # 计算当前折的索引范围
        start = j * fold_size
        end = (j + 1) * fold_size
        idx = slice(start, end)
        X_part, y_part = X[idx, :], y[idx]

        # 取第i折作为验证集
        if j == i:
            X_valid, y_valid = X_part, y_part
        else:
            X_train.append(X_part)
            y_train.append(y_part)

    # 使用 torch.cat 将训练集的各个部分拼接起来
    X_train_cat = torch.cat(X_train, dim=0)
    y_train_cat = torch.cat(y_train, dim=0)

    return X_train_cat, y_train_cat, X_valid, y_valid

# k折交叉验证训练并评估模型
def k_fold(
    k,
    X_train,
    y_train,
    num_epochs,
    learning_rate,
    weight_decay,
    batch_size,
):
    train_l_sum, valid_l_sum = 0.0, 0.0

    print(f"开始{k}折交叉验证...")
    for i in range(k):
        # 获取当前折的数据
        X_train_fold, y_train_fold, X_valid_fold, y_valid_fold = get_k_fold_data(
            k, i, X_train, y_train
        )

        # 定义并训练模型
        net = get_net()
        train_ls, valid_ls = train(
            net,
            X_train_fold,
            y_train_fold,
            X_valid_fold,
            y_valid_fold,
            num_epochs,
            learning_rate,
            weight_decay,
            batch_size,
        )

        # 累加最后一轮的误差
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]

        print(
            f"折{i + 1}，训练log rmse: {float(train_ls[-1]):f}, "
            f"验证log rmse: {float(valid_ls[-1]):f}"
        )

    return train_l_sum / k, valid_l_sum / k



if __name__ == "__main__":
    # 超参数设置
    k, num_epochs, lr, weight_decay, batch_size = 5, 100, 0.1, 25, 256

    # 进行K折交叉验证
    train_l, valid_l = k_fold(
        k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size
    )
    print(
        f"\n{k}折验证: 平均训练log rmse: {float(train_l):f}, "
        f"平均验证log rmse: {float(valid_l):f}"
    )
