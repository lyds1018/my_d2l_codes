import os

import pandas as pd
import torch

basedir = os.path.dirname(os.path.abspath(__file__))
os.chdir(basedir)


# 数据预处理
def load_and_process_data():
    # --- 1. 读取数据 ---
    train_data = pd.read_csv("./house_data/train.csv")
    test_data = pd.read_csv("./house_data/test.csv")

    # --- 2. 预处理数据 ---
    # 拼接训练集和测试集（去掉ID列和标签列）
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

    # 划分训练集和测试集
    n_train = train_data.shape[0]

    train_features = all_features[:n_train].astype("float32")
    test_features = all_features[n_train:].astype("float32")
    train_labels = train_data.SalePrice.astype("float32")

    # --- 3. 保存数据 ---
    # 保存为 CSV 文件
    train_features.to_csv("./processed_data/train_features.csv", index=False)
    test_features.to_csv("./processed_data/test_features.csv", index=False)
    train_labels.to_csv("./processed_data/train_labels.csv", index=False)

    # 保存为 pt 文件
    torch.save(
        torch.tensor(train_features.values, dtype=torch.float32),
        "./processed_data/train_features.pt",
    )
    torch.save(
        torch.tensor(test_features.values, dtype=torch.float32),
        "./processed_data/test_features.pt",
    )
    torch.save(
        torch.tensor(train_labels.values, dtype=torch.float32).reshape(-1, 1),
        "./processed_data/train_labels.pt",
    )


if __name__ == "__main__":
    load_and_process_data()
