import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import random

from groundingdino.util.inference import load_model, load_image, predict

from src.config import (
    MODEL_CONFIG_PATH,
    WEIGHTS_DIR,
    BOX_THRESHOLD,
    TEXT_THRESHOLD,
    CHESTXRAY_BBOX_CSV,
    DATASET_CONFIGS
)
from src.lora import LoRAGroundingDINO
from src.bbox_utils import (
    load_chestxray_gt_bboxes,
    load_mvtec_gt_bboxes,
    compute_iou,
    normalize_boxes_to_xyxy
)


def prepare_training_data(dataset_name, categories, k_shot=8):
    config = DATASET_CONFIGS[dataset_name]
    prompts = config.get("prompts", {})
    
    all_samples = []
    
    if dataset_name == "medical":
        gt_data = load_chestxray_gt_bboxes(CHESTXRAY_BBOX_CSV)
        chestxray_root = Path(config["root"]).parent.parent / "downloads" / "nih" / "images"
        
        for category in categories:
            category_images = {}
            for key, data in gt_data.items():
                if data["finding"] == category:
                    img_name = data["image"]
                    if img_name not in category_images:
                        category_images[img_name] = []
                    category_images[img_name].extend(data["bboxes"])
            
            valid_samples = []
            for img_name, boxes in category_images.items():
                img_path = chestxray_root / img_name
                if img_path.exists():
                    valid_samples.append({
                        "image_path": str(img_path),
                        "gt_boxes": boxes,
                        "category": category,
                        "prompt": prompts.get(category, category.lower())
                    })
            
            if len(valid_samples) > k_shot:
                selected = random.sample(valid_samples, k_shot)
            else:
                selected = valid_samples
            
            all_samples.extend(selected)
    
    elif dataset_name == "MVTec":
        for category in categories:
            gt_boxes = load_mvtec_gt_bboxes(config["root"], category)
            
            for img_path, boxes in gt_boxes.items():
                all_samples.append({
                    "image_path": img_path,
                    "gt_boxes": boxes,
                    "category": category,
                    "prompt": prompts.get(category, category.lower())
                })
        
        if len(all_samples) > k_shot * len(categories):
            all_samples = random.sample(all_samples, k_shot * len(categories))
    
    return all_samples


def compute_giou_loss(pred_boxes, gt_boxes, device):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return torch.tensor(0.0, device=device)
    
    total_loss = 0.0
    
    for gt_box in gt_boxes:
        best_iou = 0
        for pred_box in pred_boxes:
            iou = compute_iou(pred_box, gt_box)
            best_iou = max(best_iou, iou)
        total_loss += (1.0 - best_iou)
    
    return torch.tensor(total_loss / len(gt_boxes), device=device, requires_grad=True)


def train_lora(
    categories,
    dataset_name="medical",
    k_shot=8,
    num_epochs=30,
    lr=1e-4,
    rank=4,
    alpha=1.0,
    save_path="weights/lora_medical.pth"
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
    
    training_data = prepare_training_data(dataset_name, categories, k_shot)
    print(f"Training samples: {len(training_data)}")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_samples = 0
        
        random.shuffle(training_data)
        
        pbar = tqdm(training_data, desc=f"Epoch {epoch+1}/{num_epochs}")
        for sample in pbar:
            img_path = sample["image_path"]
            gt_boxes = sample["gt_boxes"]
            prompt = sample["prompt"]
            
            if not prompt.endswith("."):
                prompt = prompt + "."
            prompt = prompt.lower()
            
            try:
                image_source, image = load_image(img_path)
                image = image.to(device)
                h, w = image_source.shape[:2]
                
                boxes, logits, phrases = predict(
                    model=model.base_model,
                    image=image,
                    caption=prompt,
                    box_threshold=0.1,
                    text_threshold=0.1
                )
                
                if len(boxes) > 0:
                    pred_boxes_xyxy = normalize_boxes_to_xyxy(boxes.tolist(), w, h)
                    
                    loss = compute_giou_loss(pred_boxes_xyxy, gt_boxes, device)
                    
                    if len(logits) > 0:
                        conf_target = torch.ones_like(logits) * 0.9
                        conf_loss = nn.functional.mse_loss(logits, conf_target)
                        loss = loss + conf_loss * 0.1
                else:
                    loss = torch.tensor(1.0, device=device, requires_grad=True)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_samples += 1
                
                pbar.set_postfix({"loss": loss.item()})
                
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
    parser.add_argument("--dataset", default="medical", choices=["medical", "MVTec"])
    parser.add_argument("--k_shot", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()
    
    if args.dataset == "medical":
        categories = ["Pneumonia", "Nodule", "Effusion"]
    else:
        categories = DATASET_CONFIGS["MVTec"]["categories"]
    
    train_lora(
        categories=categories,
        dataset_name=args.dataset,
        k_shot=args.k_shot,
        num_epochs=args.epochs,
        lr=args.lr,
        rank=args.rank,
        alpha=args.alpha,
        save_path=f"weights/lora_{args.dataset}_k{args.k_shot}_r{args.rank}.pth"
    )


if __name__ == "__main__":
    main()
