import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping

basedir = os.path.dirname(os.path.abspath(__file__))
os.chdir(basedir)

# --- 加载数据 ---
X_train = pd.read_csv("./processed_data/train_features.csv")
y = pd.read_csv("./processed_data/train_labels.csv").values.ravel()

y_log = np.log1p(y)


# --- 设置XGBoost 参数 ---
XGB_PARAMS = dict(
    objective="reg:squarederror",
    eval_metric="rmse",
    learning_rate=0.01,
    max_depth=5,
    subsample=0.7,
    colsample_bytree=0.7,
    n_estimators=3000,
    random_state=42,
    n_jobs=-1,
)


# --- K 交叉验证（寻找最佳迭代次数）---
def k_fold(X, y, params, k=5, early_stopping_rounds=100):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    rmses = []
    best_iters = []

    print(f"开始 {k}-折 XGBoost 交叉验证...")

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X), start=1):
        X_tr, X_va = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_va = y[train_idx], y[valid_idx]

        model = XGBRegressor(
            **params,
            callbacks=[EarlyStopping(rounds=early_stopping_rounds, save_best=True)],
        )

        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        preds = model.predict(X_va)
        rmse = np.sqrt(mean_squared_error(y_va, preds))
        rmses.append(rmse)
        best_iters.append(model.best_iteration)

        print(f"折 {fold}: best_iter={model.best_iteration}, log RMSE={rmse:.5f}")

    avg_rmse = float(np.mean(rmses))
    avg_best_iter = int(np.mean(best_iters) * 1.1)  # 增加 10% 作为安全边际

    print("\n交叉验证结果")
    print(f"平均 log RMSE: {avg_rmse:.5f}")
    print(f"最佳迭代次数: {avg_best_iter}")

    return avg_best_iter


# --- 训练，保存模型 ---
def train_and_save(X, y, params, N, model_path):
    final_params = params.copy()
    final_params["n_estimators"] = N

    model = XGBRegressor(**final_params)
    model.fit(X, y, verbose=False)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"\n最终模型已保存至: {model_path}")
    return model


if __name__ == "__main__":
    # 找到 K 折平均的最佳迭代次数
    N = k_fold(X_train, y_log, XGB_PARAMS, k=5, early_stopping_rounds=100)

    # 训练并保存最终模型
    train_and_save(
        X_train,
        y_log,
        XGB_PARAMS,
        N=N,
        model_path="./model/xgb_model.pkl",
    )
