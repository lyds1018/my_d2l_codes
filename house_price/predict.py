import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch import nn

basedir = os.path.dirname(os.path.abspath(__file__))
os.chdir(basedir)


# --- PyTorch 神经网络模型 ---
def get_net(in_features):
    net = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 1),
    )
    return net


# --- 加载数据 ---
def load_test_data():
    # 获取 Id 列
    test_csv = pd.read_csv("./house_data/test.csv")
    test_ids = test_csv["Id"].values

    # 加载测试集
    test_features = torch.load("./processed_data/test_features.pt")
    return test_features.numpy(), test_features, test_ids


# --- 预测结果 ---
# XGBoost 预测
def predict_xgb(X_test):
    with open("./model/xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    preds_log = model.predict(X_test).flatten()

    return preds_log


# PyTorch 预测
def predict_pytorch(X_test, in_features):
    net = get_net(in_features)
    net.load_state_dict(torch.load("./model/pytorch_model.pth"))
    net.eval()
    with torch.no_grad():
        preds = net(X_test).numpy().flatten()

    return preds


# --- 加权融合 ---
def fuse_predict(weight_xgb=0.6, weight_pt=0.4):
    # 加载测试集
    xgb_X_test, pt_X_test, test_ids = load_test_data()
    in_features = pt_X_test.shape[1]

    # 加权预测结果
    xgb_preds_log = predict_xgb(xgb_X_test)
    xgb_preds = np.expm1(xgb_preds_log)
    pt_preds = predict_pytorch(pt_X_test, in_features)
    preds = weight_xgb * xgb_preds + weight_pt * pt_preds

    # 构建提交文件
    submission = pd.DataFrame({"Id": test_ids, "SalePrice": preds})
    submission.to_csv("./result/submission.csv", index=False)

    print("文件已保存到: ./result/submission.csv")


if __name__ == "__main__":
    fuse_predict(weight_xgb=0.6, weight_pt=0.4)
