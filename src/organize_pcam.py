import os
import shutil
import pandas as pd
from tqdm import tqdm

current_file_path = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(src_dir)

RAW_DIR = os.path.join(project_root, "downloads", "pcam")
IMG_DIR = os.path.join(RAW_DIR, "train")
CSV_PATH = os.path.join(RAW_DIR, "train_labels.csv")
TARGET_ROOT = os.path.join(project_root, "data", "Pathology")

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error, cannot find {CSV_PATH}")
        return

    print("reading CSV file ...")
    df = pd.read_csv(CSV_PATH)
    category = "cell"

    path_good = os.path.join(TARGET_ROOT, category, "test", "normal")
    path_bad = os.path.join(TARGET_ROOT, category, "test", category)

    os.makedirs(path_good, exist_ok=True)
    os.makedirs(path_bad, exist_ok=True)

    normal_df = df[df['label'] == 0]
    n_normals = min(len(normal_df), 2000)
    normal_ids = normal_df.sample(n=n_normals, random_state=42)['id'].values

    tumor_df = df[df['label'] == 1]
    n_tumors = min(len(tumor_df), 2000)
    tumor_ids = tumor_df.sample(n=n_tumors, random_state=42)['id'].values

    print("Organizing normal images...")
    for img_id in tqdm(normal_ids):
        filename = f"{img_id}.tif"
        src = os.path.join(IMG_DIR, filename)
        dst = os.path.join(path_good, filename)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)

    print("Organizing tumor images...")
    for img_id in tqdm(tumor_ids):
        filename = f"{img_id}.tif"
        src = os.path.join(IMG_DIR, filename)
        dst = os.path.join(path_bad, filename)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)

    print("organization finished.")

if __name__ == "__main__":
    main()