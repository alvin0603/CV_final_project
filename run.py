from tqdm import tqdm
import os
import src.config as cfg
from src.detector import GroundingDINODetector
from src.utils import get_balanced_test_images, save_image_safe, calculate_metrics_advanced, save_metrics_to_csv

def main():
    detector = GroundingDINODetector(
        config_path=cfg.MODEL_CONFIG_PATH,
        weights_dir=cfg.WEIGHTS_DIR
    )

    for dataset_name in cfg.ACTIVE_DATASETS:
        print(f"=== Processing Dataset: {dataset_name} ===")
        dataset_cfg = cfg.DATASET_CONFIGS[dataset_name]
        
        categories = dataset_cfg["categories"]
        data_root = dataset_cfg["root"]
        output_root = dataset_cfg["output"]
        prompt_map = dataset_cfg["prompts"]
        default_prompt = dataset_cfg["default_prompt"]
        normal_folder = dataset_cfg["normal_folder"]

        all_metrics = []

        for category in categories:
            prompt = prompt_map.get(category, default_prompt)
            print(f"Category: {category}")
            
            normal_imgs, anomaly_imgs = get_balanced_test_images(
                data_root, category, normal_folder, sample_num=cfg.SAMPLE_NUM_PER_CLASS
            )

            y_true = []
            y_scores = []
            
            max_conf_anomaly = 0.0
            max_conf_normal = 0.0

            imgs_to_process = [(img, 0) for img in normal_imgs] + \
                              [(img, 1) for img in anomaly_imgs]
            
            for img_path, label in tqdm(imgs_to_process, desc=f"Eval {category}"):
                try:
                    result_img, box_count, max_score = detector.run_inference(
                        img_path, prompt, cfg.BOX_THRESHOLD, cfg.TEXT_THRESHOLD
                    )
                    
                    y_true.append(label)
                    y_scores.append(max_score)

                    if label == 1:
                        max_conf_anomaly = max(max_conf_anomaly, max_score)
                    else:
                        max_conf_normal = max(max_conf_normal, max_score)

                    parent_dir = os.path.dirname(img_path)
                    defect_type = os.path.basename(parent_dir)
                    
                    name_without_ext = os.path.splitext(os.path.basename(img_path))[0]
                    filename = f"pred_{name_without_ext}.png"
                    
                    output_path = os.path.join(output_root, category, defect_type, filename)
                    save_image_safe(output_path, result_img)

                except Exception as e:
                    print(f"Error: {e}")

            ap, auc = calculate_metrics_advanced(y_true, y_scores)
            
            row = {
                "Category": category,
                "Image_AP": ap,
                "Image_AUC": auc,
                "Max_Conf_Anomaly": round(max_conf_anomaly, 4),
                "Max_Conf_Normal": round(max_conf_normal, 4)
            }
            all_metrics.append(row)

            print(f"   Stats -> AP: {ap}, AUC: {auc}")

        csv_file = save_metrics_to_csv(output_root, all_metrics)
        print(f"\nDataset {dataset_name} Evaluation Complete.")
        print(f"Metrics saved to: {csv_file}")

if __name__ == "__main__":
    main()