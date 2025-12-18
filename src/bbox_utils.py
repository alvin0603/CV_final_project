import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import glob


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def mask_to_bboxes(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    for contour in contours:
        if cv2.contourArea(contour) < 10:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        bboxes.append([x, y, x + w, y + h])
    
    return bboxes


def load_mvtec_gt_bboxes(data_root, category):
    gt_dict = {}
    category_path = Path(data_root) / category
    test_path = category_path / "test"
    gt_path = category_path / "ground_truth"
    
    if not gt_path.exists():
        return gt_dict
    
    for defect_folder in gt_path.iterdir():
        if not defect_folder.is_dir():
            continue
        
        defect_name = defect_folder.name
        test_defect_path = test_path / defect_name
        
        for mask_file in defect_folder.glob("*_mask.png"):
            img_name = mask_file.name.replace("_mask", "")
            img_path = test_defect_path / img_name
            
            if img_path.exists():
                bboxes = mask_to_bboxes(mask_file)
                if bboxes:
                    gt_dict[str(img_path)] = bboxes
    
    return gt_dict


def load_chestxray_gt_bboxes(bbox_csv_path, image_root=None):
    gt_dict = {}
    
    if not os.path.exists(bbox_csv_path):
        return gt_dict
    
    df = pd.read_csv(bbox_csv_path)
    
    for _, row in df.iterrows():
        img_name = row["Image Index"]
        finding = row["Finding Label"]
        x = float(row.iloc[2])
        y = float(row.iloc[3])
        w = float(row.iloc[4])
        h = float(row.iloc[5])
        
        bbox = [x, y, x + w, y + h]
        
        key = f"{finding}_{img_name}"
        
        if key not in gt_dict:
            gt_dict[key] = {"image": img_name, "finding": finding, "bboxes": []}
        gt_dict[key]["bboxes"].append(bbox)
    
    return gt_dict


def match_predictions_to_gt(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    if len(pred_boxes) == 0:
        return [], [], len(gt_boxes)
    
    if len(gt_boxes) == 0:
        return [], list(range(len(pred_boxes))), 0
    
    matched_preds = []
    matched_gts = set()
    fp_indices = []
    
    sorted_indices = np.argsort(pred_scores)[::-1]
    
    for pred_idx in sorted_indices:
        pred_box = pred_boxes[pred_idx]
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gts:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            matched_preds.append((pred_idx, best_gt_idx, best_iou))
            matched_gts.add(best_gt_idx)
        else:
            fp_indices.append(pred_idx)
    
    fn_count = len(gt_boxes) - len(matched_gts)
    
    return matched_preds, fp_indices, fn_count


def compute_ap(precisions, recalls):
    if len(precisions) == 0:
        return 0.0
    
    sorted_indices = np.argsort(recalls)
    recalls = np.array(recalls)[sorted_indices]
    precisions = np.array(precisions)[sorted_indices]
    
    recalls = np.concatenate([[0], recalls, [1]])
    precisions = np.concatenate([[0], precisions, [0]])
    
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    
    return ap


def compute_bbox_metrics(all_predictions, all_gt_boxes, iou_threshold=0.5):
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_iou = 0.0
    iou_count = 0
    
    all_scores = []
    all_matches = []
    total_gt = 0
    
    for img_key in set(list(all_predictions.keys()) + list(all_gt_boxes.keys())):
        pred_data = all_predictions.get(img_key, {"boxes": [], "scores": []})
        gt_data = all_gt_boxes.get(img_key, [])
        
        pred_boxes = pred_data["boxes"]
        pred_scores = pred_data["scores"]
        gt_boxes = gt_data
        
        total_gt += len(gt_boxes)
        
        matches, fp_indices, fn_count = match_predictions_to_gt(
            pred_boxes, pred_scores, gt_boxes, iou_threshold
        )
        
        total_tp += len(matches)
        total_fp += len(fp_indices)
        total_fn += fn_count
        
        for pred_idx, gt_idx, iou in matches:
            total_iou += iou
            iou_count += 1
            all_scores.append((pred_scores[pred_idx], 1))
        
        for fp_idx in fp_indices:
            all_scores.append((pred_scores[fp_idx], 0))
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    avg_iou = total_iou / iou_count if iou_count > 0 else 0.0
    
    if len(all_scores) > 0:
        all_scores.sort(key=lambda x: x[0], reverse=True)
        precisions = []
        recalls = []
        tp_cumsum = 0
        fp_cumsum = 0
        
        for score, is_tp in all_scores:
            if is_tp:
                tp_cumsum += 1
            else:
                fp_cumsum += 1
            
            p = tp_cumsum / (tp_cumsum + fp_cumsum)
            r = tp_cumsum / total_gt if total_gt > 0 else 0
            precisions.append(p)
            recalls.append(r)
        
        mAP = compute_ap(precisions, recalls)
    else:
        mAP = 0.0
    
    return {
        "mAP": mAP,
        "avg_iou": avg_iou,
        "precision": precision,
        "recall": recall,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "total_gt": total_gt
    }


def normalize_boxes_to_xyxy(boxes, img_width, img_height):
    if len(boxes) == 0:
        return []
    
    boxes_np = np.array(boxes)
    
    if boxes_np.max() <= 1.0:
        cx = boxes_np[:, 0] * img_width
        cy = boxes_np[:, 1] * img_height
        w = boxes_np[:, 2] * img_width
        h = boxes_np[:, 3] * img_height
        
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        return [[x1[i], y1[i], x2[i], y2[i]] for i in range(len(boxes_np))]
    
    return boxes_np.tolist()
