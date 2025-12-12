import os

import pandas as pd
import torch

basedir = os.path.dirname(os.path.abspath(__file__))
os.chdir(basedir)


# 数据加载与预处理
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

    # 保存列名
    feature_names = all_features.columns.tolist()

    # 将 DataFrame 转换为 NumPy 数组，再转换为 PyTorch 张量
    n_train = train_data.shape[0]

    train_features = torch.tensor(
        all_features[:n_train].values.astype("float32"), dtype=torch.float32
    )
    test_features = torch.tensor(
        all_features[n_train:].values.astype("float32"), dtype=torch.float32
    )
    train_labels = torch.tensor(
        train_data.SalePrice.values.astype("float32"), dtype=torch.float32
    ).reshape(-1, 1)

    return (
        train_features,
        test_features,
        train_labels,
        train_data,
        test_data,
        feature_names,
    )


# 保存处理后的数据
def save_processed_data(
    train_features,
    test_features,
    train_labels,
    train_data,
    test_data,
    feature_names,
):
    # 保存处理后的训练数据
    torch.save(train_features, "./processed_data/train_features.pt")
    torch.save(test_features, "./processed_data/test_features.pt")
    torch.save(train_labels, "./processed_data/train_labels.pt")

    train_features_df = pd.DataFrame(train_features.numpy(), columns=feature_names)
    test_features_df = pd.DataFrame(test_features.numpy(), columns=feature_names)

    # 保存处理后的完整数据
    train_processed = pd.concat(
        [train_data["Id"], train_features_df, train_data["SalePrice"]], axis=1
    )
    train_processed.to_csv("./processed_data/train_processed.csv", index=False)

    test_processed = pd.concat([test_data["Id"], test_features_df], axis=1)
    test_processed.to_csv("./processed_data/test_processed.csv", index=False)


if __name__ == "__main__":
    (
        train_features,
        test_features,
        train_labels,
        train_data,
        test_data,
        feature_names,
    ) = load_and_process_data()
    save_processed_data(
        train_features,
        test_features,
        train_labels,
        train_data,
        test_data,
        feature_names,
    )
