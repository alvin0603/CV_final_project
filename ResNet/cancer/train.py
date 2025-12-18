import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import copy

from dataset import CancerDataset, load_data_split
from model import CancerModel
from metrics import compute_metrics

# Configuration
IMG_ROOT = Path("/home/choulin/CV_final_project/data/cancer/train")
CSV_PATH = Path("/home/choulin/CV_final_project/data/cancer/train_labels.csv")
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4

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
        
        if (i + 1) % 100 == 0:
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Prepare Data
    train_df, val_df = load_data_split(IMG_ROOT, CSV_PATH)
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),    
    ])
    
    train_ds = CancerDataset(train_df, transform=train_transform)
    val_ds = CancerDataset(val_df, transform=val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 2. Model, Loss, Optimizer
    model = CancerModel(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_auc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    # 3. Training Loop
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_ap, val_auc = evaluate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AP: {val_ap:.4f} | Val AUC: {val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), "ResNet/cancer/best_model.pth")
            print("Saved new best model.")
            
    print(f"Best Validation AUC: {best_auc:.4f}")

if __name__ == "__main__":
    main()
