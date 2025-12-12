# dataset.py
import os
from pathlib import Path
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]
NUM_CLASSES = len(CLASSES)


def find_image(img_root, filename):
    """在 images_001 ~ images_012 自動搜尋圖檔"""
    for i in range(1, 13):
        p = img_root / f"images_{i:03d}" / "images" / filename
        if p.exists():
            return str(p)
    return None


def load_split_lists(img_root, csv_path, train_list, test_list):
    df = pd.read_csv(csv_path)

    # 生成 mapping（filename → label row）
    df = df.rename(columns={"Image Index": "filename", "Finding Labels": "labels"})
    df = df.set_index("filename")

    with open(train_list, "r") as f:
        train_files = f.read().splitlines()
    with open(test_list, "r") as f:
        test_files = f.read().splitlines()

    # 加完整路徑
    train_paths_raw = [find_image(img_root, x) for x in train_files]
    train_paths = [p for p in train_paths_raw if p is not None]

    test_paths_raw = [find_image(img_root, x) for x in test_files]
    test_paths = [p for p in test_paths_raw if p is not None]

    # 再從 train 中分出 val
    train_paths, val_paths = train_test_split(train_paths, test_size=0.1, random_state=42)

    return train_paths, val_paths, test_paths, df


class CXRDataset(Dataset):
    def __init__(self, img_paths, label_df, transform=None):
        self.paths = img_paths
        self.label_df = label_df
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")

        filename = os.path.basename(img_path)
        findings = self.label_df.loc[filename]["labels"]

        # binary Normal / Abnormal
        binary_label = 0.0 if findings == "No Finding" else 1.0

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(binary_label).float(), findings
