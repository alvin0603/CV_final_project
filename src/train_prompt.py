import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import cv2
import random

from groundingdino.util.inference import load_model, load_image

from src.config import (
    MODEL_CONFIG_PATH,
    WEIGHTS_DIR,
    BOX_THRESHOLD,
    CHESTXRAY_BBOX_CSV,
    DATASET_CONFIGS
)
from src.prompt_tuning import PromptTuner, save_prompt_tuner
from src.bbox_utils import (
    load_chestxray_gt_bboxes,
    load_mvtec_gt_bboxes,
    compute_iou
)


class FewShotDataset(Dataset):
    def __init__(self, image_paths, gt_boxes, category, prompt):
        self.image_paths = image_paths
        self.gt_boxes = gt_boxes
        self.category = category
        self.prompt = prompt
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image_source, image = load_image(img_path)
        gt = self.gt_boxes.get(img_path, [])
        
        return {
            "image": image,
            "image_source": image_source,
            "gt_boxes": gt,
            "category": self.category,
            "prompt": self.prompt,
            "image_path": img_path
        }


def compute_detection_loss(pred_boxes, pred_logits, gt_boxes, device):
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    gt_tensor = torch.tensor(gt_boxes, dtype=torch.float32, device=device)
    
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    for gt_box in gt_tensor:
        ious = []
        for pred_box in pred_boxes:
            iou = compute_iou(
                pred_box.detach().cpu().tolist(),
                gt_box.cpu().tolist()
            )
            ious.append(iou)
        
        if len(ious) > 0:
            max_iou = max(ious)
            best_idx = ious.index(max_iou)
            iou_loss = 1.0 - max_iou
            conf_loss = (1.0 - pred_logits[best_idx]).pow(2)
            total_loss = total_loss + iou_loss + conf_loss
    
    return total_loss / len(gt_tensor)


def prepare_few_shot_data(dataset_name, category, k_shot=4):
    config = DATASET_CONFIGS[dataset_name]
    
    if dataset_name == "medical":
        gt_data = load_chestxray_gt_bboxes(CHESTXRAY_BBOX_CSV)
        
        category_images = {}
        for key, data in gt_data.items():
            if data["finding"] == category:
                img_name = data["image"]
                if img_name not in category_images:
                    category_images[img_name] = []
                category_images[img_name].extend(data["bboxes"])
        
        chestxray_root = Path(config["root"]).parent.parent / "downloads" / "nih" / "images"
        
        valid_samples = []
        gt_boxes = {}
        for img_name, boxes in category_images.items():
            img_path = chestxray_root / img_name
            if img_path.exists():
                valid_samples.append(str(img_path))
                gt_boxes[str(img_path)] = boxes
        
        if len(valid_samples) > k_shot:
            selected = random.sample(valid_samples, k_shot)
        else:
            selected = valid_samples
        
        selected_gt = {k: gt_boxes[k] for k in selected}
        
        return selected, selected_gt
    
    elif dataset_name == "MVTec":
        gt_boxes = load_mvtec_gt_bboxes(config["root"], category)
        
        all_images = list(gt_boxes.keys())
        if len(all_images) > k_shot:
            selected = random.sample(all_images, k_shot)
        else:
            selected = all_images
        
        selected_gt = {k: gt_boxes[k] for k in selected}
        
        return selected, selected_gt
    
    return [], {}


def train_prompt_tuning(
    categories,
    dataset_name="medical",
    k_shot=4,
    num_epochs=20,
    lr=1e-3,
    save_path="weights/prompt_tuned.pth"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    weights_path = os.path.join(WEIGHTS_DIR, "groundingdino_swint_ogc.pth")
    model = load_model(MODEL_CONFIG_PATH, weights_path)
    model.eval()
    model.to(device)
    
    config = DATASET_CONFIGS[dataset_name]
    prompts = config.get("prompts", {})
    
    prompt_tuner = PromptTuner(
        embed_dim=256,
        num_prompt_tokens=8,
        categories=categories
    ).to(device)
    
    optimizer = torch.optim.AdamW(prompt_tuner.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_samples = 0
        
        for category in categories:
            images, gt_boxes = prepare_few_shot_data(dataset_name, category, k_shot)
            
            if len(images) == 0:
                print(f"No data for {category}")
                continue
            
            prompt = prompts.get(category, category.lower())
            if not prompt.endswith("."):
                prompt = prompt + "."
            prompt = prompt.lower()
            
            for img_path in images:
                image_source, image = load_image(img_path)
                image = image.to(device)
                h, w = image_source.shape[:2]
                
                tokenized = model.tokenizer(
                    [prompt], padding="longest", return_tensors="pt"
                ).to(device)
                
                with torch.no_grad():
                    from groundingdino.models.GroundingDINO.groundingdino import (
                        generate_masks_with_special_tokens_and_transfer_map
                    )
                    
                    (
                        text_self_attention_masks,
                        position_ids,
                        _,
                    ) = generate_masks_with_special_tokens_and_transfer_map(
                        tokenized, model.specical_tokens, model.tokenizer
                    )
                    
                    if model.sub_sentence_present:
                        tokenized_for_encoder = {
                            k: v for k, v in tokenized.items() if k != "attention_mask"
                        }
                        tokenized_for_encoder["attention_mask"] = text_self_attention_masks
                        tokenized_for_encoder["position_ids"] = position_ids
                    else:
                        tokenized_for_encoder = tokenized
                    
                    bert_output = model.bert(**tokenized_for_encoder)
                    encoded_text = model.feat_map(bert_output["last_hidden_state"])
                
                enhanced_text = prompt_tuner(encoded_text, category)
                
                loss = torch.mean((enhanced_text - encoded_text).pow(2)) * 0.1
                
                gt = gt_boxes.get(img_path, [])
                if len(gt) > 0:
                    target_response = torch.ones(1, device=device) * 0.9
                    current_response = enhanced_text.mean()
                    detection_loss = (current_response - target_response).pow(2)
                    loss = loss + detection_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_samples += 1
        
        scheduler.step()
        
        avg_loss = total_loss / max(num_samples, 1)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_prompt_tuner(prompt_tuner, save_path)
    print(f"Saved prompt tuner to {save_path}")
    
    return prompt_tuner


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="medical", choices=["medical", "MVTec"])
    parser.add_argument("--k_shot", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    if args.dataset == "medical":
        categories = ["Pneumonia", "Nodule", "Effusion"]
    else:
        categories = DATASET_CONFIGS["MVTec"]["categories"]
    
    train_prompt_tuning(
        categories=categories,
        dataset_name=args.dataset,
        k_shot=args.k_shot,
        num_epochs=args.epochs,
        lr=args.lr,
        save_path=f"weights/prompt_tuned_{args.dataset}_k{args.k_shot}.pth"
    )


if __name__ == "__main__":
    main()
