import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path

from dataset import CXRDataset, load_split_lists, CLASSES
from model import CXRModel
from sklearn.metrics import roc_auc_score, average_precision_score

IMG_ROOT = Path("/home/choulin/CV_final_project/data/ChestXray")
CSV_PATH = IMG_ROOT / "Data_Entry_2017.csv"
TRAIN_LIST = IMG_ROOT / "train_val_list.txt"
TEST_LIST = IMG_ROOT / "test_list.txt"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, _, test_paths, df = load_split_lists(
        IMG_ROOT, CSV_PATH, TRAIN_LIST, TEST_LIST
    )

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    test_ds = CXRDataset(test_paths, df, transform=tf)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = CXRModel(num_classes=1)
    # Note: ensure you load the correct checkpoint (best_auc_model.pth)
    try:
        model.load_state_dict(torch.load("best_auc_model.pth", map_location=device))
    except FileNotFoundError:
        print("Model file not found. Please train first.")
        return

    model = model.to(device)
    model.eval()

    # Collect all predictions and metadata
    all_probs = []
    all_findings = []
    
    # We iterate freely; labels in dataset are binary, but we need the 'findings' string.
    print("Running inference on test set...")
    with torch.no_grad():
        for imgs, _, findings in test_loader:
            imgs = imgs.to(device)
            logits, _ = model(imgs) # logits: [B, 1]
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
            all_probs.extend(probs)
            # findings is a tuple of strings (batch_size,)
            all_findings.extend(findings)
            
    all_probs = np.array(all_probs)
    all_findings = np.array(all_findings)
    
    # Prepare CSV output
    results_list = []
    
    # Iterate over each disease class to compute metrics
    # Compare Class vs "No Finding"
    # Normal (Negative) class: "No Finding"
    # Anomaly (Positive) class: The specific Disease
    
    # Identify indices for "No Finding"
    all_indices = np.arange(len(all_findings))
    no_finding_mask = (all_findings == "No Finding")
    no_finding_indices = all_indices[no_finding_mask]
    
    # Sample 3000 "No Finding" cases (or less if not enough)
    # Mimic src/organize_xray.py: n_normals = min(len(normal_df), 3000); sample(..., random_state=42)
    n_normals = min(len(no_finding_indices), 3000)
    if n_normals > 0:
        np.random.seed(42)
        # usage of choice without replacement
        sampled_no_finding_indices = np.random.choice(no_finding_indices, size=n_normals, replace=False)
    else:
        sampled_no_finding_indices = np.array([], dtype=int)

    print(f"Total No Finding: {len(no_finding_indices)}, Sampled: {len(sampled_no_finding_indices)}")
    
    # Overall Binary Metrics (Sick vs Not Sick) - Note: This might need to be on the FULL set or the Sampled set?
    # Usually "Overall" implies full test set. But the user asked to "do sampling like organize_xray.py", 
    # which was for creating a balanced(ish) dataset for validation/testing.
    # The user request "evaluate... do same sampling" implies the metrics for the table should be based on this sampling.
    
    # Let's calculate Overall AUC on the FULL set for reference, but the Table rows (Categories) will use the sampling.
    binary_ground_truth = [(0 if f == "No Finding" else 1) for f in all_findings]
    overall_auc = roc_auc_score(binary_ground_truth, all_probs)
    print(f"Overall Binary AUC (Sick vs Not Sick) [Full Test Set]: {overall_auc:.4f}")

    # Target categories as per prompts_test/level_1/X-ray.csv
    TARGET_CATEGORIES = ["Pneumonia", "Nodule", "Effusion", "Infiltration"]

    for disease in TARGET_CATEGORIES:
        # Find indices where 'disease' is present
        disease_indices = [i for i, f in enumerate(all_findings) if disease in f.split('|')]
        
        if len(disease_indices) == 0:
            print(f"No samples for {disease}")
            # Add placeholders
            results_list.append({
                "Category": disease,
                "Image_AP": 0.0,
                "Image_AUC": 0.0,
                "Max_Conf_Anomaly": 0.0,
                "Max_Conf_Normal": 0.0
            })
            continue
            
        # Combine Disease (All) and No Finding (Sampled) indices
        # We use the fixed sampled set for negatives
        target_indices = np.concatenate([disease_indices, sampled_no_finding_indices]).astype(int)
        
        # Filter predictions and create binary labels for this specific comparison
        # Positive (1): Has Disease
        # Negative (0): No Finding (Sampled)
        
        sub_probs = all_probs[target_indices]
        # Create labels: 1 if index in disease_indices, 0 if in sampled_no_finding_indices
        # Note: 'disease_indices' and 'sampled_no_finding_indices' are disjoint because one is "No Finding" 
        # and the other has the disease.
        
        sub_labels = []
        # Optimization: we know the structure of target_indices: [disease_part, no_finding_part]
        # But let's be safe
        disease_set = set(disease_indices)
        for idx in target_indices:
            if idx in disease_set:
                sub_labels.append(1)
            else:
                sub_labels.append(0)
        
        sub_labels = np.array(sub_labels)
        
        if len(np.unique(sub_labels)) < 2:
             # Can't calculate AUC if only one class
             category_auc = 0.5
             category_ap = 0.0
             max_conf_anom = 0.0
             max_conf_norm = 0.0
        else:
            category_auc = roc_auc_score(sub_labels, sub_probs)
            category_ap = average_precision_score(sub_labels, sub_probs)
            
            max_conf_anom = sub_probs[sub_labels == 1].mean() if np.any(sub_labels == 1) else 0.0
            max_conf_norm = sub_probs[sub_labels == 0].mean() if np.any(sub_labels == 0) else 0.0
        
        results_list.append({
            "Category": disease,
            "Image_AP": category_ap,
            "Image_AUC": category_auc,
            "Max_Conf_Anomaly": max_conf_anom,
            "Max_Conf_Normal": max_conf_norm
        })

    # Save to CSV
    import pandas as pd
    res_df = pd.DataFrame(results_list)
    output_csv = "X-ray.csv"
    res_df.to_csv(output_csv, index=False, float_format='%.4f')
    print(f"\nSaved per-class results to {output_csv}")
    print(res_df)


if __name__ == "__main__":
    main()
