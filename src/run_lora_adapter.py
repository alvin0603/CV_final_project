import os
import torch
import csv
from pathlib import Path
from tqdm import tqdm
import glob

from groundingdino.util.inference import load_model, load_image

from src.config import (
    MODEL_CONFIG_PATH,
    WEIGHTS_DIR,
    DATASET_CONFIGS
)
from src.lora_adapter import LoRAAdapterGroundingDINO
from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate_lora_adapter(
    dataset_name,
    weights_path,
    output_dir,
    lora_rank=4,
    lora_alpha=1.0,
    adapter_bottleneck=64
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_weights = os.path.join(WEIGHTS_DIR, "groundingdino_swint_ogc.pth")
    base_model = load_model(MODEL_CONFIG_PATH, model_weights)
    base_model.to(device)
    
    model = LoRAAdapterGroundingDINO(
        base_model, 
        lora_rank=lora_rank, 
        lora_alpha=lora_alpha,
        adapter_bottleneck=adapter_bottleneck
    )
    model.load_hybrid(weights_path, device)
    model.eval()
    model.to(device)
    
    config = DATASET_CONFIGS[dataset_name]
    root_dir = config["root"]
    categories = config["categories"]
    prompts = config.get("prompts", {})
    normal_folder = config.get("normal_folder", "good")
    
    all_results = []
    
    for category in categories:
        print(f"\n=== LoRA+Adapter Eval: {category} ===")
        
        prompt_text = prompts.get(category, category.lower())
        if not prompt_text.endswith("."):
            prompt_text = prompt_text + "."
        prompt_text = prompt_text.lower()
        
        test_dir = os.path.join(root_dir, category, "test")
        
        if not os.path.exists(test_dir):
            print(f"  Test dir not found: {test_dir}")
            continue
        
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
        
        normal_path = os.path.join(test_dir, normal_folder)
        normal_images = []
        if os.path.exists(normal_path):
            for ext in extensions:
                normal_images.extend(glob.glob(os.path.join(normal_path, ext)))
        
        all_subfolders = [d for d in os.listdir(test_dir) 
                        if os.path.isdir(os.path.join(test_dir, d))]
        anomaly_folders = [d for d in all_subfolders if d != normal_folder]
        
        anomaly_images = []
        for f in anomaly_folders:
            f_path = os.path.join(test_dir, f)
            for ext in extensions:
                anomaly_images.extend(glob.glob(os.path.join(f_path, ext)))
        
        y_true = []
        y_scores = []
        normal_confs = []
        anomaly_confs = []
        
        for img_path in tqdm(normal_images, desc=f"Eval {category} (Normal)"):
            try:
                image_source, image = load_image(img_path)
                image = image.to(device)
                
                with torch.no_grad():
                    from groundingdino.util.inference import preprocess_caption
                    caption = preprocess_caption(prompt_text)
                    outputs = model.base_model(image[None], captions=[caption])
                    logits = outputs["pred_logits"].sigmoid()[0]
                    max_conf = logits.max().item()
                
                y_true.append(0)
                y_scores.append(max_conf)
                normal_confs.append(max_conf)
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        for img_path in tqdm(anomaly_images, desc=f"Eval {category} (Anomaly)"):
            try:
                image_source, image = load_image(img_path)
                image = image.to(device)
                
                with torch.no_grad():
                    from groundingdino.util.inference import preprocess_caption
                    caption = preprocess_caption(prompt_text)
                    outputs = model.base_model(image[None], captions=[caption])
                    logits = outputs["pred_logits"].sigmoid()[0]
                    max_conf = logits.max().item()
                
                y_true.append(1)
                y_scores.append(max_conf)
                anomaly_confs.append(max_conf)
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        if len(set(y_true)) < 2:
            print(f"  Skipping {category}: not enough classes")
            continue
        
        try:
            auc = roc_auc_score(y_true, y_scores)
            ap = average_precision_score(y_true, y_scores)
        except:
            auc = 0.0
            ap = 0.0
        
        max_conf_normal = max(normal_confs) if normal_confs else 0.0
        max_conf_anomaly = max(anomaly_confs) if anomaly_confs else 0.0
        
        print(f"  Image_AUC: {auc:.4f} | Image_AP: {ap:.4f} | Max_Conf_Anomaly: {max_conf_anomaly:.4f} | Max_Conf_Normal: {max_conf_normal:.4f}")
        
        all_results.append({
            "Category": category,
            "Image_AP": round(ap, 4),
            "Image_AUC": round(auc, 4),
            "Max_Conf_Anomaly": round(max_conf_anomaly, 4),
            "Max_Conf_Normal": round(max_conf_normal, 4)
        })
    
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "image_level_metrics_lora_adapter.csv")
    
    headers = ["Category", "Image_AP", "Image_AUC", "Max_Conf_Anomaly", "Max_Conf_Normal"]
    
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)
    
    print(f"\nResults saved to {csv_path}")
    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="medical", choices=["medical", "MVTec", "Pathology"])
    parser.add_argument("--weights", required=True, help="Path to LoRA+Adapter weights")
    parser.add_argument("--output_dir", default="outputs/image_level/lora_adapter")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--adapter_bottleneck", type=int, default=64)
    args = parser.parse_args()
    
    evaluate_lora_adapter(
        dataset_name=args.dataset,
        weights_path=args.weights,
        output_dir=args.output_dir,
        lora_rank=args.lora_rank,
        adapter_bottleneck=args.adapter_bottleneck
    )


if __name__ == "__main__":
    main()
