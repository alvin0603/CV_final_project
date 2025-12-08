import os
import glob
import random
import cv2
import csv
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

def get_balanced_test_images(data_dir, category, normal_folder_name, sample_num=5):
    test_root = os.path.join(data_dir, category, "test")
    
    normal_path = os.path.join(test_root, normal_folder_name)
    all_subfolders = [d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d))]
    anomaly_folders = [d for d in all_subfolders if d != normal_folder_name]

    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    
    normal_images = []
    if os.path.exists(normal_path):
        for ext in extensions:
            normal_images.extend(glob.glob(os.path.join(normal_path, ext)))
    
    anomaly_images = []
    for f in anomaly_folders:
        f_path = os.path.join(test_root, f)
        for ext in extensions:
            anomaly_images.extend(glob.glob(os.path.join(f_path, ext)))

    selected_normal = []
    if normal_images:
        count = min(len(normal_images), sample_num)
        selected_normal = random.sample(normal_images, count)

    selected_anomaly = []
    if anomaly_images:
        count = min(len(anomaly_images), sample_num)
        selected_anomaly = random.sample(anomaly_images, count)

    return selected_normal, selected_anomaly

def save_image_safe(output_path, image):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        is_success, im_buf_arr = cv2.imencode(".png", image)
        if is_success:
            im_buf_arr.tofile(output_path)
            return True
    except Exception:
        return False
    return False

def calculate_metrics_advanced(y_true, y_scores, threshold=0.35):
    try:
        ap = average_precision_score(y_true, y_scores)
    except Exception:
        ap = 0.0

    try:
        auc = roc_auc_score(y_true, y_scores)
    except Exception:
        auc = 0.5
        
    y_pred = [1 if score >= threshold else 0 for score in y_scores]
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
        
    return round(ap, 4), round(auc, 4), round(acc, 4), round(f1, 4), round(precision, 4), round(recall, 4)

def save_metrics_to_csv(output_dir, metrics_data):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "evaluation_metrics.csv")
    
    headers = ["Category", "Image_AP", "Image_AUC", "Accuracy", "F1_Score", "Precision", "Recall", "Max_Conf_Anomaly", "Max_Conf_Normal"]
    
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in metrics_data:
            writer.writerow(row)
    
    return csv_path