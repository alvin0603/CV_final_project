import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_mvtec_data(root_dir, test_size=0.2, random_state=42):
    """
    Loads MVTec data for binary classification.
    Merges 'train' and 'test' sets, labels 'good' as 0 and others as 1.
    Returns train and val DataFrames.
    """
    root_path = Path(root_dir)
    data = []
    
    # Categories are subdirectories in root_dir
    # e.g. bottle, cable, capsule...
    # Exclude files like readme.txt
    categories = [d for d in root_path.iterdir() if d.is_dir()]
    
    for category_path in categories:
        category_name = category_path.name
        
        # Walk through all files in this category
        for file_path in category_path.rglob("*"):
             if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
                # Determine label based on parent folder name
                # Structure: category / train|test / type / image.png
                # 'type' is the parent folder of the image
                parent_name = file_path.parent.name
                
                if parent_name == 'good':
                    label = 0
                else:
                    label = 1
                
                data.append({
                    "path": str(file_path),
                    "label": label,
                    "category": category_name,
                    "split": file_path.parent.parent.name # train or test (just for info)
                })
    
    df = pd.DataFrame(data)
    
    # Stratified split based on label (to ensure both normal and anomaly in train/val)
    # We ignore the original train/test split because original train has no anomalies
    train_df, val_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df['label']
    )
    
    return train_df, val_df

class MVTecDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = row['path']
        label = row['label']
        
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # return a dummy image to avoid crashing, but ideally should filter
            img = Image.new('RGB', (224, 224))
        
        if self.transform:
            img = self.transform(img)
            
        # Binary classification target: float for BCE loss
        return img, torch.tensor([label], dtype=torch.float32)
