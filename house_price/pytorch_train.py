import os

import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

basedir = os.path.dirname(os.path.abspath(__file__))
os.chdir(basedir)

# --- 加载数据 ---
X_train = torch.load("./processed_data/train_features.pt")
y_train = torch.load("./processed_data/train_labels.pt")


# --- 定义模型、评价指标和优化器 ---
loss = nn.MSELoss()
in_features = X_train.shape[1]


# 模型
def get_net():
    net = nn.Sequential(
        nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 1)
    )
    return net


# 评价指标：对数均方根误差
def log_rmse(net, X, y):
    preds = net(X)
    rmse = torch.sqrt(loss(torch.log1p(preds), torch.log1p(y)))
    return rmse.item()


# 优化器与训练过程
def train(
    net,
    X_train,
    y_train,
    X_test,
    y_test,
    num_epochs,
    learning_rate,
    weight_decay,
    batch_size,
):
    # 创建数据迭代器
    train_dataset = TensorDataset(X_train, y_train)
    train_iter = DataLoader(train_dataset, batch_size, shuffle=True)

    # 使用Adam优化器
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
            train_ls.append(log_rmse(net, X_train, y_train))
            # 计算验证/测试集上的 log_rmse
            if y_test is not None:
                test_ls.append(log_rmse(net, X_test, y_test))

    return train_ls, test_ls


# --- 超参数调优（k折交叉验证）---
def k_fold(
    k,
    X_train,
    y_train,
    num_epochs,
    learning_rate,
    weight_decay,
    batch_size,
):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    train_l_sum, valid_l_sum = 0.0, 0.0

    print(f"开始{k}折交叉验证...")
    for i in range(k):
        # 获取数据
        train_idx, valid_idx = list(kf.split(X_train))[i]
        X_train_fold, y_train_fold = X_train[train_idx], y_train[train_idx]
        X_valid_fold, y_valid_fold = X_train[valid_idx], y_train[valid_idx]

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


# --- 训练，保存模型 ---
def train_and_save(
    train_features,
    train_labels,
    num_epochs,
    lr,
    weight_decay,
    batch_size,
):
    print("训练模型...")
    net = get_net()
    # 在全部训练集上进行训练
    train_ls, _ = train(
        net,
        train_features,
        train_labels,
        None,  # 无验证集
        None,  # 无验证集标签
        num_epochs,
        lr,
        weight_decay,
        batch_size,
    )

    final_train_rmse = float(train_ls[-1])
    print(f"最终训练log rmse：{final_train_rmse:f}")

    # 保存模型参数
    torch.save(net.state_dict(), "./model/pytorch_model.pth")
    print("模型已保存到 ./model/pytorch_model.pth")


if __name__ == "__main__":
    # 超参数设置
    k, num_epochs, lr, weight_decay, batch_size = 5, 100, 0.1, 25, 256

    """
    # 进行K折交叉验证
    train_l, valid_l = k_fold(
        k, X_train, y_train, num_epochs, lr, weight_decay, batch_size
    )
    print(
        f"\n{k}折验证: 平均训练log rmse: {float(train_l):f}, "
        f"平均验证log rmse: {float(valid_l):f}"
    )
    """

    # 训练并保存最终模型
    train_and_save(
        X_train,
        y_train,
        num_epochs,
        lr,
        weight_decay,
        batch_size,
    )
