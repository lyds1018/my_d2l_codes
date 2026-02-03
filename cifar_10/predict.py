import os

import numpy as np
import pandas as pd
import torch
from data_set import ImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms as T
from train import ResNet18

torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
basedir = os.path.dirname(os.path.abspath(__file__))

test_transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

train_dataset = ImageDataset(
    csv_file=os.path.join(basedir, "data", "train.csv"),
    img_path=os.path.join(basedir, "data"),
)

test_dataset = ImageDataset(
    csv_file=os.path.join(basedir, "data", "sampleSubmission.csv"),
    img_path=os.path.join(basedir, "data", "test"),
    transform=test_transform,
    index=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=512,
    shuffle=False,
    num_workers=12,
    pin_memory=True,
    persistent_workers=True,
)

model = ResNet18().to(device)
model.load_state_dict(torch.load(os.path.join(basedir, "model", "best_model.pth")))
model.eval()
model.requires_grad_(False)

preds = np.empty(len(test_dataset), dtype=np.int64)
ptr = 0

with torch.no_grad():
    with torch.amp.autocast("cuda"):
        for images in test_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)

            bs = predicted.size(0)
            preds[ptr : ptr + bs] = predicted.cpu().numpy()
            ptr += bs

idx2label = {v: k for k, v in train_dataset.label2idx.items()}
pred_labels = [idx2label[i] for i in preds]

test_df = pd.read_csv(os.path.join(basedir, "data", "sampleSubmission.csv"))
submission_df = pd.DataFrame(
    {test_df.columns[0]: test_df.iloc[:, 0], "label": pred_labels}
)

submission_df.to_csv(os.path.join(basedir, "result", "submission.csv"), index=False)

print("预测结果已保存到 /result/submission.csv 文件中")
