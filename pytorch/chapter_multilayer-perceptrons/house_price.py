import os

import pandas as pd
import torch
from d2l import torch as d2l
from torch import nn

# 1.读取数据
basedir = os.path.dirname(os.path.abspath(__file__))
os.chdir(basedir)

train_data = pd.read_csv("./house_data/train.csv")
test_data = pd.read_csv("./house_data/test.csv")

# 2.预处理数据
# 拼接训练集和测试集（去掉ID列）
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# 标准化数值列
numeric_features = all_features.dtypes[all_features.dtypes != "object"].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std())
)
# 填充缺失值（标准化后均值为 0）
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 对所有离散特征做独热编码
all_features = pd.get_dummies(all_features, dummy_na=True)

# 转换为张量
n_train = train_data.shape[0]

train_features = torch.tensor(
    all_features[:n_train].values.astype(float), dtype=torch.float32
)
test_features = torch.tensor(
    all_features[n_train:].values.astype(float), dtype=torch.float32
)
train_labels = torch.tensor(
    train_data.SalePrice.values.astype(float), dtype=torch.float32
).reshape(-1, 1)


# 3.定义模型、评价指标和优化器
loss = nn.MSELoss()
in_features = train_features.shape[1]


# 模型
def get_net():
    net = nn.Sequential(nn.Linear(in_features, 1))
    return net


# 评价指标
def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float("inf"))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()


# 优化器
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
    train_iter = d2l.load_array((train_features, train_labels), batch_size)

    # 使用Adam优化器
    # weight_decay参数实现权重衰减（L2正则化）
    optimizer = torch.optim.Adam(
        net.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # 训练过程
    train_ls, test_ls = [], []
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()

        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))

    return train_ls, test_ls


# 4.超参数调优（k折交叉验证）
# 获取第i折交叉验证所需要的训练和验证数据
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None

    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        # 取第i折作为验证集
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], dim=0)
            y_train = torch.cat([y_train, y_part], dim=0)

    return X_train, y_train, X_valid, y_valid


# k折交叉验证训练并评估模型
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0

    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(
            net, *data, num_epochs, learning_rate, weight_decay, batch_size
        )
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]

        print(
            f"折{i + 1}，训练log rmse{float(train_ls[-1]):f}, "
            f"验证log rmse{float(valid_ls[-1]):f}"
        )

    return train_l_sum / k, valid_l_sum / k


# 调优过程
k, num_epochs, lr, weight_decay, batch_size = 5, 128, 30, 1e-2, 128
"""
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print(f'{k}折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')
"""


# 5.训练，导出
def train_and_pred(
    train_features,
    test_features,
    train_labels,
    test_data,
    num_epochs,
    lr,
    weight_decay,
    batch_size,
):
    net = get_net()
    train_ls, _ = train(
        net,
        train_features,
        train_labels,
        None,
        None,
        num_epochs,
        lr,
        weight_decay,
        batch_size,
    )
    print(f"训练log rmse：{float(train_ls[-1]):f}")
    # 将网络应用于测试集。
    preds = net(test_features).detach().numpy()
    # 将其重新格式化以导出到Kaggle
    test_data["SalePrice"] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data["Id"], test_data["SalePrice"]], axis=1)
    submission.to_csv("submission.csv", index=False)


train_and_pred(
    train_features,
    test_features,
    train_labels,
    test_data,
    num_epochs,
    lr,
    weight_decay,
    batch_size,
)
