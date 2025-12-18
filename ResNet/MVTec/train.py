import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import copy
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

from dataset import MVTecDataset, load_mvtec_data
from model import MVTecResNet

# Configuration
IMG_ROOT = Path("/home/choulin/CV_final_project/data/MVTec")
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4

def compute_metrics(labels, probs):
    try:
        if len(np.unique(labels)) < 2:
            return 0.0, 0.5
        auc = roc_auc_score(labels, probs)
        ap = average_precision_score(labels, probs)
    except:
        auc = 0.5
        ap = 0.0
    return ap, auc

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for i, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        if (i + 1) % 50 == 0:
            print(f"Batch {i+1}/{len(loader)}, Loss: {loss.item():.4f}")
            
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            probs = torch.sigmoid(outputs)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    epoch_loss = running_loss / len(loader.dataset)
    ap, auc = compute_metrics(all_labels, all_probs)
    
    return epoch_loss, ap, auc

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Prepare Data
    print(f"Loading data from {IMG_ROOT}...")
    train_df, val_df = load_mvtec_data(IMG_ROOT)
    print(f"Train samples: {len(train_df)} (Normal: {len(train_df[train_df['label']==0])}, Anomaly: {len(train_df[train_df['label']==1])})")
    print(f"Val samples: {len(val_df)} (Normal: {len(val_df[val_df['label']==0])}, Anomaly: {len(val_df[val_df['label']==1])})")
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),    
    ])
    
    train_ds = MVTecDataset(train_df, transform=train_transform)
    val_ds = MVTecDataset(val_df, transform=val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 2. Model, Loss, Optimizer
    model = MVTecResNet(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_auc = 0.0
    
    # 3. Training Loop
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_ap, val_auc = evaluate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AP: {val_ap:.4f} | Val AUC: {val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            # Save relative to the script location to handle different CWDs
            save_path = Path(__file__).parent / "best_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model to {save_path}")
            
    print(f"Best Validation AUC: {best_auc:.4f}")

if __name__ == "__main__":
    main()
