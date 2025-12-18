import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

basedir = os.path.dirname(os.path.abspath(__file__))
os.chdir(basedir)


class ImageDataset(Dataset):
    def __init__(self, csv_file, img_path, transform=None, indexed=False):
        self.df = pd.read_csv(csv_file)
        self.img_path = img_path
        self.transform = transform
        self.indexed = indexed

        # label 编码
        if not indexed:
            labels = sorted(self.df["label"].unique())
            self.label2idx = {label: i for i, label in enumerate(labels)}
            self.df["label_idx"] = self.df["label"].map(self.label2idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_path, self.df.iloc[idx]["image"])
        if not self.indexed:
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
