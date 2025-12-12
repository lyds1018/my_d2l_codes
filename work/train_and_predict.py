import os

import k_fold_validation
import pandas as pd
import torch
from torch import nn

basedir = os.path.dirname(os.path.abspath(__file__))
os.chdir(basedir)

# --- 加载处理好的数据 ---
train_features = torch.load("./processed_data/train_features.pt")
train_labels = torch.load("./processed_data/train_labels.pt")
test_features = torch.load("./processed_data/test_features.pt")


# --- 定义模型、评价指标和优化器 ---
loss = nn.MSELoss()
in_features = train_features.shape[1]

# 模型
get_net = k_fold_validation.get_net

# 评价指标：对数均方根误差
log_rmse = k_fold_validation.log_rmse

# 优化器与训练过程
train = k_fold_validation.train


# --- 训练，导出预测结果 ---
def train_and_pred(
    train_features,
    test_features,
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

    # 在测试集上进行预测
    net.eval()  # 设置为评估模式
    with torch.no_grad():
        preds = net(test_features).numpy()

    # 导出为 submission.csv
    test_data = pd.read_csv("./house_data/test.csv")
    submission = pd.concat(
        [test_data["Id"], pd.DataFrame({"SalePrice": preds.reshape(-1)})], axis=1
    )
    submission.to_csv("./result/submission.csv", index=False)


if __name__ == "__main__":
    # 超参数设置
    num_epochs, lr, weight_decay, batch_size = 100, 0.1, 25, 256

    # 训练并导出预测结果
    train_and_pred(
        train_features,
        test_features,
        train_labels,
        num_epochs,
        lr,
        weight_decay,
        batch_size,
    )
