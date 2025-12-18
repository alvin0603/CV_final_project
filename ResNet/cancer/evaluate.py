import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import numpy as np
import pandas as pd

from dataset import CancerDataset, load_data_split
from model import CancerModel
from metrics import compute_metrics

# Configuration
IMG_ROOT = Path("/home/choulin/CV_final_project/data/cancer/train")
CSV_PATH = Path("/home/choulin/CV_final_project/data/cancer/train_labels.csv")
BATCH_SIZE = 32

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Re-use the same split logic to get validation set
    _, val_df = load_data_split(IMG_ROOT, CSV_PATH)
    
    # Sampling logic mimicking organize_pcam.py
    # organize_pcam.py samples up to 2000 normal and 2000 tumor lines
    normal_df = val_df[val_df['label'] == 0]
    tumor_df = val_df[val_df['label'] == 1]

    n_normals = min(len(normal_df), 2000)
    sampled_normals = normal_df.sample(n=n_normals, random_state=42)

    n_tumors = min(len(tumor_df), 2000)
    sampled_tumors = tumor_df.sample(n=n_tumors, random_state=42)

    val_df = pd.concat([sampled_normals, sampled_tumors])
    print(f"Evaluating on sampled set: {len(val_df)} samples ({len(sampled_normals)} normal, {len(sampled_tumors)} tumor)")
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    val_ds = CancerDataset(val_df, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    model = CancerModel(pretrained=False) # Load structure only
    
    # Resolve path relative to this script to handle running from different directories
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "best_model.pth")
    # Also support if running from root with original hardcoded path just in case, but preferred determines relative to script
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        # Fallback to checking relative path if script logic differs
        try: 
             fallback_path = "ResNet/cancer/best_model.pth"
             model.load_state_dict(torch.load(fallback_path, map_location=device))
             print(f"Loaded model from {fallback_path}")
        except FileNotFoundError:
             print(f"Model file not found at {model_path} or {fallback_path}. Please train first.")
             return
        
    model = model.to(device)
    model.eval()
    
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    # Metrics
    ap, auc = compute_metrics(all_labels, all_probs)
    
    all_probs = np.array(all_probs).flatten()
    max_conf_anom = np.max(all_probs)
    
    # Max probability attributed to the normal class (which is 1 - prob_anomaly)
    # The user asked for "Max Conf (Norm.)", which usually means the highest confidence assigned to a normal sample prediction.
    # If the model predicts p for anomaly (class 1), then probability for normal (class 0) is 1-p.
    max_conf_norm = np.max(1 - all_probs)

    print("\n===== EVALUATION RESULTS =====")
    print(f"Image AP: {ap:.4f}")
    print(f"Image AUC: {auc:.4f}")
    print(f"Max Conf (Anom.): {max_conf_anom:.4f}")
    print(f"Max Conf (Norm.): {max_conf_norm:.4f}")
    print("==============================")
    
    # Save to CSV
    output_csv = os.path.join(current_dir, "results.csv")
    # import pandas as pd (Removed)
    
    results_df = pd.DataFrame([{
        "Category": "Cancer",
        "Image_AP": f"{ap:.4f}",
        "Image_AUC": f"{auc:.4f}",
        "Max_Conf_Anomaly": f"{max_conf_anom:.4f}",
        "Max_Conf_Normal": f"{max_conf_norm:.4f}"
    }])
    
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    main()
