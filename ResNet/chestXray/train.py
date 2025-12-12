# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import numpy as np

from dataset import load_split_lists, CXRDataset
from model import CXRModel
# from metrics import compute_binary_metrics

# ================================
# 絕對路徑（直接指向資料夾）
# ================================
IMG_ROOT = Path("/home/choulin/CV_final_project/data/ChestXray")
CSV_PATH = IMG_ROOT / "Data_Entry_2017.csv"
TRAIN_LIST = IMG_ROOT / "train_val_list.txt"
TEST_LIST = IMG_ROOT / "test_list.txt"

PATIENCE = 20   # early stopping patience


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 讀取 train / val / test split
    train_paths, val_paths, test_paths, df = load_split_lists(
        IMG_ROOT, CSV_PATH, TRAIN_LIST, TEST_LIST
    )

    # transforms
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_ds = CXRDataset(train_paths, df, transform=train_tf)
    val_ds = CXRDataset(val_paths, df, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

    model = CXRModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ================================
    # Early Stopping variables
    # ================================
    best_auc = -np.inf
    epochs_no_improve = 0

    # ================================
    # Training Loop
    # ================================
    for epoch in range(100):  # 最長 100 epoch，會因 early stopping 提前結束
        model.train()
        train_loss = 0.0
        for imgs, labels, _ in train_loader:
            labels = labels.unsqueeze(1) # [B] -> [B, 1]
            imgs, labels = imgs.to(device), labels.to(device)

            logits, _ = model(imgs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)

        # ================================
        # Validation
        # ================================
        model.eval()
        model.eval()
        val_loss = 0.0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                labels = labels.unsqueeze(1)
                
                logits, anom_score = model(imgs) # anom_score is sigmoid(logits) if from model.py
                # Note: model.py returns (logits, anomaly_score). anomaly_score is max(sigmoid(logits)).
                # For binary (1 class), logits [B, 1]. sigmoid(logits) [B, 1]. max(dim=1) is trivial.
                # simpler:
                probs = torch.sigmoid(logits)
                
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            val_auc = 0.5
            
        print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val AUC: {val_auc:.4f}")

        # ================================
        # Early Stopping Logic (based on AUC)
        # ================================
        if val_auc > best_auc:
            best_auc = val_auc
            epochs_no_improve = 0

            torch.save(model.state_dict(), "best_auc_model.pth")
            print(f"New best AUC: {best_auc:.4f} — model saved!")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs")

        if epochs_no_improve >= PATIENCE:
            print(f"Early Stopping triggered at epoch {epoch}")
            break

    print(f"Training complete! Best AUC = {best_auc:.4f}")
    print("Best model saved to best_auc_model.pth")


if __name__ == "__main__":
    main()
