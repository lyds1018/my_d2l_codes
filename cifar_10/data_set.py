import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, csv_file, img_path, transform=None, index=True):
        self.df = pd.read_csv(csv_file)
        self.img_path = img_path
        self.transform = transform  # 是否图像增强
        self.index = index  # 是否需要标签

        # label 编码
        if self.index:
            labels = sorted(self.df["label"].unique())
            self.label2idx = {
                label: i for i, label in enumerate(labels)
            }  # 构建字典，i 从 0 开始
            self.df["label_idx"] = self.df["label"].map(self.label2idx)  # 字典映射

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 图片文件夹 + 图片文件名
        img_path = os.path.join(self.img_path, f"{self.df.iloc[idx]['id']}.png")

        if self.index:
            label = self.df.iloc[idx]["label_idx"]
            image = Image.open(img_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            return image, torch.tensor(label, dtype=torch.long)

        else:
            image = Image.open(img_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            return image
