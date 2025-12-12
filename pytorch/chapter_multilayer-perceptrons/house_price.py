import os

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# --- 1. 读取数据 ---
basedir = os.path.dirname(os.path.abspath(__file__))
os.chdir(basedir)

train_data = pd.read_csv("./house_data/train.csv")
test_data = pd.read_csv("./house_data/test.csv")


# --- 2. 预处理数据 ---
# 拼接训练集和测试集（去掉ID列和训练集的 SalePrice列）
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# 标准化数值列
numeric_features = all_features.dtypes[all_features.dtypes != "object"].index
# 标准化 (x - mean) / std
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std())
)
# 填充缺失值（标准化后均值为 0）
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 对所有离散特征做独热编码
all_features = pd.get_dummies(all_features, dummy_na=True)

# 转换为张量
n_train = train_data.shape[0]

# 将 DataFrame 转换为 NumPy 数组，再转换为 PyTorch 张量
train_features = torch.tensor(
    all_features[:n_train].values.astype("float32"), dtype=torch.float32
)
test_features = torch.tensor(
    all_features[n_train:].values.astype("float32"), dtype=torch.float32
)
train_labels = torch.tensor(
    train_data.SalePrice.values.astype("float32"), dtype=torch.float32
).reshape(-1, 1)


# --- 3. 定义模型、评价指标和优化器 ---
loss = nn.MSELoss()
in_features = train_features.shape[1]


# 模型：线性回归模型
def get_net():
    # 单层全连接网络，输入维度为 in_features，输出维度为 1
    net = nn.Sequential(nn.Linear(in_features, 1))
    return net


# 评价指标：对数均方根误差
def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    # 预测值
    preds = net(features)
    # 截断，确保所有值 >= 1
    clipped_preds = torch.clamp(preds, 1.0, float("inf"))

    # 计算 log(y) 和 log(y_hat) 之间的 MSE 的平方根
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()


# 优化器与训练过程
def train(
    net: nn.Module,
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


# --- 4. 超参数调优（k折交叉验证）---
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

    print(f"\nStarting {k}-fold cross-validation...")
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


# --- 5. 训练，导出预测结果 ---
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
    print("\nStarting final training and prediction...")
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

    # 在测试集上进行预测
    net.eval()  # 设置为评估模式
    with torch.no_grad():
        preds = net(test_features).cpu().numpy()  # 预测，移到CPU，转为NumPy

    # 将预测值重新格式化并添加到测试数据的副本中
    # 注意：我们预测的是 SalePrice，但评估指标是 log(SalePrice) 的 RMSE。
    # 理论上，为了与 log_rmse 评估一致，预测输出应该是 log(Price)，但在您的原始代码中，
    # 模型的输出是直接对 price 的预测，并且使用 MSE Loss。
    # 导出时，直接使用模型输出作为 SalePrice (Price)。
    test_data_with_preds = test_data.copy()
    test_data_with_preds["SalePrice"] = pd.Series(preds.flatten())

    # 导出为 submission.csv
    submission = test_data_with_preds[["Id", "SalePrice"]]
    submission.to_csv("submission.csv", index=False)
    print("Prediction results saved to submission.csv")


if __name__ == "__main__":
    # 超参数设置
    k, num_epochs, lr, weight_decay, batch_size = 5, 128, 30, 1e-2, 128

    # --- K折交叉验证 (用于调参) ---
    # train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
    # print(f'\n{k}折验证: 平均训练log rmse: {float(train_l):f}, '
    #       f'平均验证log rmse: {float(valid_l):f}')

    # --- 最终训练和预测 ---
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
