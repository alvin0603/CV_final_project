import os
import shutil
import pandas as pd
from tqdm import tqdm

current_file_path = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(src_dir)

RAW_DIR = os.path.join(project_root, "downloads", "nih")
IMG_DIR = os.path.join(RAW_DIR, "images")
CSV_PATH = os.path.join(RAW_DIR, "Data_Entry_2017.csv")
TARGET_ROOT = os.path.join(project_root, "data", "medical")

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error, cannot find {CSV_PATH}")
        return

    print("reading CSV file ...")
    df = pd.read_csv(CSV_PATH)
    target_diseases = ["Pneumonia", "Nodule", "Effusion", "Infiltration"]
    
    normal_df = df[df['Finding Labels'] == "No Finding"]
    n_normals = min(len(normal_df), 3000)
    normal_imgs = normal_df.sample(n=n_normals, random_state=42)['Image Index'].values

    for disease in target_diseases:
        print(f"Organizing category: {disease} ...")

        path_good = os.path.join(TARGET_ROOT, disease, "test", "good")
        path_bad = os.path.join(TARGET_ROOT, disease, "test", disease)
        
        os.makedirs(path_good, exist_ok=True)
        os.makedirs(path_bad, exist_ok=True)

        for img_name in tqdm(normal_imgs):
            src = os.path.join(IMG_DIR, img_name)
            dst = os.path.join(path_good, img_name)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy(src, dst)

        disease_df = df[df['Finding Labels'].str.contains(disease)]
        for img_name in tqdm(disease_df['Image Index'].values):
            src = os.path.join(IMG_DIR, img_name)
            dst = os.path.join(path_bad, img_name)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy(src, dst)

    print("organization finished.")

if __name__ == "__main__":
    main()