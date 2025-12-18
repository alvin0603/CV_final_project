import os
import torch
import csv
from tqdm import tqdm
from pathlib import Path
from groundingdino.util.inference import load_model, load_image, predict

from src.config import (
    MODEL_CONFIG_PATH,
    WEIGHTS_DIR,
    BOX_THRESHOLD,
    TEXT_THRESHOLD,
    IOU_THRESHOLD,
    CHESTXRAY_BBOX_CSV,
    DATASET_CONFIGS,
    BBOX_ACTIVE_DATASETS,
    PROMPT_LEVEL
)
from src.bbox_utils import (
    load_mvtec_gt_bboxes,
    load_chestxray_gt_bboxes,
    compute_bbox_metrics,
    normalize_boxes_to_xyxy
)


def find_chestxray_image(img_root, img_name):
    direct_path = Path(img_root) / img_name
    if direct_path.exists():
        return str(direct_path)
    
    for i in range(1, 13):
        p = Path(img_root).parent / f"images_{i:03d}" / "images" / img_name
        if p.exists():
            return str(p)
    return None


def evaluate_mvtec_bbox(model, config, output_dir):
    data_root = config["root"]
    categories = config["categories"]
    prompts = config["prompts"]
    default_prompt = config.get("default_prompt", "defect")
    
    all_results = []
    
    for category in categories:
        print(f"\n=== MVTec Category: {category} ===")
        prompt_text = prompts.get(category, default_prompt)
        
        gt_bboxes = load_mvtec_gt_bboxes(data_root, category)
        
        if len(gt_bboxes) == 0:
            print(f"  No GT bboxes found for {category}")
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
        
        print(f"  Found GT for {len(gt_bboxes)} images")
        
        all_predictions = {}
        all_gt = {}
        
        for img_path, gt_boxes in tqdm(gt_bboxes.items(), desc=f"Eval {category}"):
            if not os.path.exists(img_path):
                continue
            
            image_source, image = load_image(img_path)
            h, w = image_source.shape[:2]
            
            boxes, logits, phrases = predict(
                model=model,
                image=image,
                caption=prompt_text,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD
            )
            
            pred_boxes = normalize_boxes_to_xyxy(boxes.tolist(), w, h) if len(boxes) > 0 else []
            pred_scores = logits.tolist() if len(logits) > 0 else []
            
            all_predictions[img_path] = {"boxes": pred_boxes, "scores": pred_scores}
            all_gt[img_path] = gt_boxes
        
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
    
    save_bbox_results(output_dir, all_results)
    return all_results


def evaluate_chestxray_bbox(model, config, output_dir):
    categories = config["categories"]
    prompts = config["prompts"]
    default_prompt = config.get("default_prompt", "disease")
    
    chestxray_root = Path(config["root"]).parent.parent / "downloads" / "nih" / "images"
    
    gt_data = load_chestxray_gt_bboxes(CHESTXRAY_BBOX_CSV)
    
    if len(gt_data) == 0:
        print("No ChestXray GT bboxes found")
        return []
    
    gt_by_finding = {}
    for key, data in gt_data.items():
        finding = data["finding"]
        if finding not in gt_by_finding:
            gt_by_finding[finding] = {}
        img_name = data["image"]
        if img_name not in gt_by_finding[finding]:
            gt_by_finding[finding][img_name] = []
        gt_by_finding[finding][img_name].extend(data["bboxes"])
    
    all_results = []
    
    category_mapping = {
        "Pneumonia": "Pneumonia",
        "Nodule": "Nodule", 
        "Effusion": "Effusion",
        "Infiltration": "Infiltration"
    }
    
    for category in categories:
        print(f"\n=== ChestXray Category: {category} ===")
        prompt_text = prompts.get(category, default_prompt)
        
        gt_finding = category_mapping.get(category, category)
        
        if gt_finding not in gt_by_finding:
            print(f"  No GT bboxes found for {category}")
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
        
        category_gt = gt_by_finding[gt_finding]
        print(f"  Found GT for {len(category_gt)} images")
        
        all_predictions = {}
        all_gt = {}
        
        for img_name, gt_boxes in tqdm(category_gt.items(), desc=f"Eval {category}"):
            img_path = find_chestxray_image(chestxray_root, img_name)
            if img_path is None:
                continue
            
            image_source, image = load_image(img_path)
            h, w = image_source.shape[:2]
            
            boxes, logits, phrases = predict(
                model=model,
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
    
    save_bbox_results(output_dir, all_results)
    return all_results


def save_bbox_results(output_dir, results):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "bbox_metrics.csv")
    
    headers = ["Category", "mAP", "Avg_IoU", "Precision", "Recall", "TP", "FP", "FN", "Total_GT"]
    
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    
    print(f"\nResults saved to {csv_path}")


def main():
    weights_path = os.path.join(WEIGHTS_DIR, "groundingdino_swint_ogc.pth")
    if not os.path.exists(weights_path):
        print(f"Weights not found at {weights_path}")
        return

    model = load_model(MODEL_CONFIG_PATH, weights_path)
    
    for dataset_name in BBOX_ACTIVE_DATASETS:
        print(f"\n{'='*50}")
        print(f"BBox Evaluation: {dataset_name}")
        print(f"{'='*50}")
        
        config = DATASET_CONFIGS[dataset_name]
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(config["output"])), 
            "bbox",
            f"{dataset_name}_Level{PROMPT_LEVEL}"
        )
        
        if dataset_name == "MVTec":
            evaluate_mvtec_bbox(model, config, output_dir)
        elif dataset_name == "medical":
            evaluate_chestxray_bbox(model, config, output_dir)


if __name__ == "__main__":
    main()
