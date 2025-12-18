import os
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import random
import glob

from groundingdino.util.inference import load_model, load_image, predict

from src.config import (
    MODEL_CONFIG_PATH,
    WEIGHTS_DIR,
    BOX_THRESHOLD,
    TEXT_THRESHOLD,
    DATASET_CONFIGS
)
from src.lora import LoRAGroundingDINO


def prepare_image_level_data(dataset_name, categories, k_shot=8, use_train=True):
    config = DATASET_CONFIGS[dataset_name]
    root_dir = config["root"]
    prompts = config.get("prompts", {})
    normal_folder = config.get("normal_folder", "good")
    
    all_samples = []
    
    for category in categories:
        if use_train:
            split_dir = os.path.join(root_dir, category, "train")
        else:
            split_dir = os.path.join(root_dir, category, "test")
        
        if not os.path.exists(split_dir):
            split_dir = os.path.join(root_dir, category, "test")
        
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
        
        normal_path = os.path.join(split_dir, normal_folder)
        normal_images = []
        if os.path.exists(normal_path):
            for ext in extensions:
                normal_images.extend(glob.glob(os.path.join(normal_path, ext)))
        
        all_subfolders = []
        if os.path.exists(split_dir):
            all_subfolders = [d for d in os.listdir(split_dir) 
                            if os.path.isdir(os.path.join(split_dir, d))]
        anomaly_folders = [d for d in all_subfolders if d != normal_folder]
        
        anomaly_images = []
        for f in anomaly_folders:
            f_path = os.path.join(split_dir, f)
            for ext in extensions:
                anomaly_images.extend(glob.glob(os.path.join(f_path, ext)))
        
        prompt = prompts.get(category, category.lower())
        
        for img_path in normal_images:
            all_samples.append({
                "image_path": img_path,
                "label": 0,
                "category": category,
                "prompt": prompt
            })
        
        for img_path in anomaly_images:
            all_samples.append({
                "image_path": img_path,
                "label": 1,
                "category": category,
                "prompt": prompt
            })
    
    normal_samples = [s for s in all_samples if s["label"] == 0]
    anomaly_samples = [s for s in all_samples if s["label"] == 1]
    
    if len(normal_samples) > k_shot:
        normal_samples = random.sample(normal_samples, k_shot)
    if len(anomaly_samples) > k_shot:
        anomaly_samples = random.sample(anomaly_samples, k_shot)
    
    return normal_samples + anomaly_samples


def train_lora_image_level(
    categories,
    dataset_name="medical",
    k_shot=8,
    num_epochs=30,
    lr=1e-4,
    rank=4,
    alpha=1.0,
    margin=0.3,
    save_path="weights/lora_image_level.pth"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    weights_path = os.path.join(WEIGHTS_DIR, "groundingdino_swint_ogc.pth")
    base_model = load_model(MODEL_CONFIG_PATH, weights_path)
    base_model.to(device)
    
    model = LoRAGroundingDINO(base_model, rank=rank, alpha=alpha)
    model.to(device)
    
    trainable_params = model.get_trainable_parameters()
    print(f"Trainable LoRA parameters: {sum(p.numel() for p in trainable_params)}")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    training_data = prepare_image_level_data(dataset_name, categories, k_shot)
    print(f"Training samples: {len(training_data)} (Normal: {sum(1 for s in training_data if s['label']==0)}, Anomaly: {sum(1 for s in training_data if s['label']==1)})")
    
    for epoch in range(num_epochs):
        model.train()
        for param in model.get_trainable_parameters():
            param.requires_grad = True
        
        total_loss = 0.0
        num_samples = 0
        
        random.shuffle(training_data)
        
        pbar = tqdm(training_data, desc=f"Epoch {epoch+1}/{num_epochs}")
        for sample in pbar:
            img_path = sample["image_path"]
            label = sample["label"]
            prompt = sample["prompt"]
            
            if not prompt.endswith("."):
                prompt = prompt + "."
            prompt = prompt.lower()
            
            try:
                image_source, image = load_image(img_path)
                image = image.to(device)
                
                from groundingdino.util.inference import preprocess_caption
                from groundingdino.util.utils import get_phrases_from_posmap
                
                caption = preprocess_caption(prompt)
                
                outputs = model.base_model(image[None], captions=[caption])
                
                logits = outputs["pred_logits"].sigmoid()[0]
                boxes = outputs["pred_boxes"][0]
                
                max_logits_per_box = logits.max(dim=1)[0]
                
                if max_logits_per_box.numel() > 0:
                    max_conf = max_logits_per_box.max()
                else:
                    max_conf = logits.mean() * 0  
                
                if label == 0:
                    loss = torch.relu(max_conf - margin) + max_conf * 0.1
                else:
                    loss = torch.relu(0.8 - max_conf) + (1.0 - max_conf) * 0.1
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_samples += 1
                
                pbar.set_postfix({"loss": f"{loss.item():.3f}", "conf": f"{max_conf.item():.3f}", "lbl": label})
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        scheduler.step()
        
        avg_loss = total_loss / max(num_samples, 1)
        print(f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save_lora(save_path)
    print(f"Saved LoRA weights to {save_path}")
    
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="medical", choices=["medical", "MVTec", "Pathology"])
    parser.add_argument("--k_shot", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=0.3, help="Margin for normal samples")
    args = parser.parse_args()
    
    if args.dataset == "medical":
        categories = ["Pneumonia", "Nodule", "Effusion", "Infiltration"]
    elif args.dataset == "MVTec":
        categories = DATASET_CONFIGS["MVTec"]["categories"]
    else:
        categories = DATASET_CONFIGS["Pathology"]["categories"]
    
    train_lora_image_level(
        categories=categories,
        dataset_name=args.dataset,
        k_shot=args.k_shot,
        num_epochs=args.epochs,
        lr=args.lr,
        rank=args.rank,
        alpha=args.alpha,
        margin=args.margin,
        save_path=f"weights/lora_image_{args.dataset}_k{args.k_shot}_r{args.rank}.pth"
    )


if __name__ == "__main__":
    main()
