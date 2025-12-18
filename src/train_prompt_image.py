import os
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import random
import glob

from groundingdino.util.inference import load_model, load_image

from src.config import (
    MODEL_CONFIG_PATH,
    WEIGHTS_DIR,
    DATASET_CONFIGS
)
from src.prompt_tuning import PromptTuner, save_prompt_tuner


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


def train_prompt_image_level(
    categories,
    dataset_name="medical",
    k_shot=8,
    num_epochs=50,
    lr=1e-3,
    margin=0.3,
    save_path="weights/prompt_image_level.pth"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    weights_path = os.path.join(WEIGHTS_DIR, "groundingdino_swint_ogc.pth")
    model = load_model(MODEL_CONFIG_PATH, weights_path)
    model.eval()
    model.to(device)
    
    for param in model.parameters():
        param.requires_grad = False
    
    prompt_tuner = PromptTuner(
        embed_dim=256,
        num_prompt_tokens=8,
        categories=categories
    ).to(device)
    
    optimizer = torch.optim.AdamW(prompt_tuner.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    training_data = prepare_image_level_data(dataset_name, categories, k_shot)
    print(f"Training samples: {len(training_data)} (Normal: {sum(1 for s in training_data if s['label']==0)}, Anomaly: {sum(1 for s in training_data if s['label']==1)})")
    
    for epoch in range(num_epochs):
        prompt_tuner.train()
        total_loss = 0.0
        num_samples = 0
        
        random.shuffle(training_data)
        
        pbar = tqdm(training_data, desc=f"Epoch {epoch+1}/{num_epochs}")
        for sample in pbar:
            img_path = sample["image_path"]
            label = sample["label"]
            category = sample["category"]
            prompt = sample["prompt"]
            
            if not prompt.endswith("."):
                prompt = prompt + "."
            prompt = prompt.lower()
            
            try:
                image_source, image = load_image(img_path)
                image = image.to(device)
                
                from groundingdino.util.inference import preprocess_caption
                
                caption = preprocess_caption(prompt)
                
                with torch.no_grad():
                    tokenized = model.tokenizer(
                        [caption], padding="longest", return_tensors="pt"
                    ).to(device)
                    
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
                
                text_diff = (enhanced_text - encoded_text).pow(2).mean()
                
                if label == 0:
                    loss = text_diff + margin
                else:
                    loss = -text_diff + 1.0
                
                loss = torch.relu(loss)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(prompt_tuner.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_samples += 1
                
                pbar.set_postfix({"loss": f"{loss.item():.3f}", "lbl": label})
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        scheduler.step()
        
        avg_loss = total_loss / max(num_samples, 1)
        print(f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_prompt_tuner(prompt_tuner, save_path)
    print(f"Saved prompt tuner to {save_path}")
    
    return prompt_tuner


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="medical", choices=["medical", "MVTec", "Pathology"])
    parser.add_argument("--k_shot", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.3)
    args = parser.parse_args()
    
    if args.dataset == "medical":
        categories = ["Pneumonia", "Nodule", "Effusion", "Infiltration"]
    elif args.dataset == "MVTec":
        categories = DATASET_CONFIGS["MVTec"]["categories"]
    else:
        categories = DATASET_CONFIGS["Pathology"]["categories"]
    
    train_prompt_image_level(
        categories=categories,
        dataset_name=args.dataset,
        k_shot=args.k_shot,
        num_epochs=args.epochs,
        lr=args.lr,
        margin=args.margin,
        save_path=f"weights/prompt_image_{args.dataset}_k{args.k_shot}.pth"
    )


if __name__ == "__main__":
    main()
