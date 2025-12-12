import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from pathlib import Path

CLASSES = ["Normal", "Anomaly"]

def load_data_split(img_root, csv_path, test_size=0.2, random_state=42):
    """
    Loads labels from CSV and splits into train/val.
    """
    df = pd.read_csv(csv_path)
    # Expected columns: id, label
    
    # We need to find the full path for each image 'id'
    # The 'id' in csv is the filename without extension (based on previous observations of hash-like ids)
    # But files in data/cancer/train have .tif extension.
    # Let's assume id match filename stem.
    
    all_files = list(Path(img_root).glob("*.tif"))
    file_map = {f.stem: f for f in all_files}
    
    valid_data = []
    
    for _, row in df.iterrows():
        img_id = row['id']
        label = row['label']
        if img_id in file_map:
            valid_data.append({
                "path": str(file_map[img_id]),
                "label": label,
                "id": img_id
            })
            
    dataset_df = pd.DataFrame(valid_data)
    
    train_df, val_df = train_test_split(dataset_df, test_size=test_size, random_state=random_state, stratify=dataset_df['label'])
    
    return train_df, val_df

class CancerDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = row['path']
        label = row['label']
        
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
            
        # Binary classification target: float for BCE loss
        return img, torch.tensor([label], dtype=torch.float32)
