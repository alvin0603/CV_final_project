import os
import torch
from tqdm import tqdm
from groundingdino.util.inference import load_model, load_image, predict, annotate

from src.config import (
    MODEL_CONFIG_PATH,
    WEIGHTS_DIR,
    BOX_THRESHOLD,
    TEXT_THRESHOLD,
    DATASET_CONFIGS,
    ACTIVE_DATASETS,
    SAMPLE_NUM_PER_CLASS
)
from src.utils import (
    get_balanced_test_images,
    save_image_safe,
    calculate_metrics_advanced,
    save_metrics_to_csv
)

def main():
    weights_path = os.path.join(WEIGHTS_DIR, "groundingdino_swint_ogc.pth")
    if not os.path.exists(weights_path):
        print(f"Weights not found at {weights_path}")
        return

    model = load_model(MODEL_CONFIG_PATH, weights_path)

    for dataset_name in ACTIVE_DATASETS:
        print(f"=== Processing Dataset: {dataset_name} ===")
        config = DATASET_CONFIGS[dataset_name]
        
        root_dir = config["root"]
        output_root = config["output"]
        categories = config["categories"]
        prompts = config["prompts"]
        normal_folder = config["normal_folder"]
        default_prompt = config.get("default_prompt", "anomaly")

        all_metrics = []

        for category in categories:
            print(f"Category: {category}")
            prompt_text = prompts.get(category, default_prompt)
            
            normal_imgs, anomaly_imgs = get_balanced_test_images(
                root_dir, category, normal_folder, int(SAMPLE_NUM_PER_CLASS)
            )

            y_true = []
            y_scores = []
            max_conf_normal = 0.0
            max_conf_anomaly = 0.0

            img_output_dir = os.path.join(output_root, category)
            os.makedirs(img_output_dir, exist_ok=True)

            for img_path in tqdm(normal_imgs, desc=f"Eval {category} (Normal)"):
                image_source, image = load_image(img_path)
                boxes, logits, phrases = predict(
                    model=model,
                    image=image,
                    caption=prompt_text,
                    box_threshold=BOX_THRESHOLD,
                    text_threshold=TEXT_THRESHOLD
                )
                
                conf = logits.max().item() if len(logits) > 0 else 0.0
                y_true.append(0)
                y_scores.append(conf)
                
                if conf > max_conf_normal:
                    max_conf_normal = conf

                filename = os.path.basename(img_path)
                annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
                save_image_safe(os.path.join(img_output_dir, "good", filename), annotated_frame)

            for img_path in tqdm(anomaly_imgs, desc=f"Eval {category} (Anomaly)"):
                image_source, image = load_image(img_path)
                boxes, logits, phrases = predict(
                    model=model,
                    image=image,
                    caption=prompt_text,
                    box_threshold=BOX_THRESHOLD,
                    text_threshold=TEXT_THRESHOLD
                )
                
                conf = logits.max().item() if len(logits) > 0 else 0.0
                y_true.append(1)
                y_scores.append(conf)

                if conf > max_conf_anomaly:
                    max_conf_anomaly = conf

                filename = os.path.basename(img_path)
                annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
                save_image_safe(os.path.join(img_output_dir, category, filename), annotated_frame)

            ap, auc, acc, f1, precision, recall = calculate_metrics_advanced(
                y_true, y_scores, threshold=BOX_THRESHOLD
            )
            
            print(f"Category: {category} | AUC: {auc} | AP: {ap} | Acc: {acc} | F1: {f1}")

            all_metrics.append({
                "Category": category,
                "Image_AP": ap,
                "Image_AUC": auc,
                "Accuracy": acc,
                "F1_Score": f1,
                "Precision": precision,
                "Recall": recall,
                "Max_Conf_Anomaly": max_conf_anomaly,
                "Max_Conf_Normal": max_conf_normal
            })

        save_metrics_to_csv(output_root, all_metrics)

if __name__ == "__main__":
    main()