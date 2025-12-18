import os
import torch
import csv
from pathlib import Path
from tqdm import tqdm

from groundingdino.util.inference import load_model, load_image, predict

from src.config import (
    MODEL_CONFIG_PATH,
    WEIGHTS_DIR,
    BOX_THRESHOLD,
    TEXT_THRESHOLD,
    IOU_THRESHOLD,
    CHESTXRAY_BBOX_CSV,
    DATASET_CONFIGS
)
from src.lora import LoRAGroundingDINO
from src.bbox_utils import (
    load_chestxray_gt_bboxes,
    compute_bbox_metrics,
    normalize_boxes_to_xyxy
)


def find_chestxray_image(img_root, img_name):
    direct_path = Path(img_root) / img_name
    if direct_path.exists():
        return str(direct_path)
    return None


def evaluate_lora(
    dataset_name,
    lora_path,
    output_dir,
    rank=4,
    alpha=1.0
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    weights_path = os.path.join(WEIGHTS_DIR, "groundingdino_swint_ogc.pth")
    base_model = load_model(MODEL_CONFIG_PATH, weights_path)
    base_model.to(device)
    
    model = LoRAGroundingDINO(base_model, rank=rank, alpha=alpha)
    model.load_lora(lora_path, device)
    model.eval()
    model.to(device)
    
    config = DATASET_CONFIGS[dataset_name]
    categories = config["categories"]
    prompts = config.get("prompts", {})
    
    all_results = []
    
    if dataset_name == "medical":
        gt_data = load_chestxray_gt_bboxes(CHESTXRAY_BBOX_CSV)
        chestxray_root = Path(config["root"]).parent.parent / "downloads" / "nih" / "images"
        
        gt_by_finding = {}
        for key, data in gt_data.items():
            finding = data["finding"]
            if finding not in gt_by_finding:
                gt_by_finding[finding] = {}
            img_name = data["image"]
            if img_name not in gt_by_finding[finding]:
                gt_by_finding[finding][img_name] = []
            gt_by_finding[finding][img_name].extend(data["bboxes"])
        
        for category in categories:
            print(f"\n=== LoRA Eval: {category} ===")
            prompt_text = prompts.get(category, category.lower())
            if not prompt_text.endswith("."):
                prompt_text = prompt_text + "."
            prompt_text = prompt_text.lower()
            
            if category not in gt_by_finding:
                print(f"  No GT for {category}")
                all_results.append({
                    "Category": category,
                    "mAP": 0.0,
                    "Avg_IoU": 0.0,
                    "Precision": 0.0,
                    "Recall": 0.0,
                    "TP": 0,
                    "FP": 0,
                    "FN": 0,
                    "Total_GT": 0
                })
                continue
            
            category_gt = gt_by_finding[category]
            
            all_predictions = {}
            all_gt = {}
            
            for img_name, gt_boxes in tqdm(category_gt.items(), desc=f"Eval {category}"):
                img_path = find_chestxray_image(chestxray_root, img_name)
                if img_path is None:
                    continue
                
                image_source, image = load_image(img_path)
                image = image.to(device)
                h, w = image_source.shape[:2]
                
                with torch.no_grad():
                    boxes, logits, phrases = predict(
                        model=model.base_model,
                        image=image,
                        caption=prompt_text,
                        box_threshold=BOX_THRESHOLD,
                        text_threshold=TEXT_THRESHOLD
                    )
                
                pred_boxes = normalize_boxes_to_xyxy(boxes.tolist(), w, h) if len(boxes) > 0 else []
                pred_scores = logits.tolist() if len(logits) > 0 else []
                
                all_predictions[img_name] = {"boxes": pred_boxes, "scores": pred_scores}
                all_gt[img_name] = gt_boxes
            
            metrics = compute_bbox_metrics(all_predictions, all_gt, IOU_THRESHOLD)
            
            print(f"  mAP: {metrics['mAP']:.4f} | IoU: {metrics['avg_iou']:.4f} | P: {metrics['precision']:.4f} | R: {metrics['recall']:.4f}")
            
            all_results.append({
                "Category": category,
                "mAP": round(metrics["mAP"], 4),
                "Avg_IoU": round(metrics["avg_iou"], 4),
                "Precision": round(metrics["precision"], 4),
                "Recall": round(metrics["recall"], 4),
                "TP": metrics["tp"],
                "FP": metrics["fp"],
                "FN": metrics["fn"],
                "Total_GT": metrics["total_gt"]
            })
    
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "bbox_metrics_lora.csv")
    
    headers = ["Category", "mAP", "Avg_IoU", "Precision", "Recall", "TP", "FP", "FN", "Total_GT"]
    
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
    parser.add_argument("--dataset", default="medical", choices=["medical", "MVTec"])
    parser.add_argument("--lora_path", required=True, help="Path to LoRA weights")
    parser.add_argument("--output_dir", default="outputs/bbox/lora")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()
    
    evaluate_lora(
        dataset_name=args.dataset,
        lora_path=args.lora_path,
        output_dir=args.output_dir,
        rank=args.rank,
        alpha=args.alpha
    )


if __name__ == "__main__":
    main()
