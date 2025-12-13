import os
import glob
import csv
import random
import shutil

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'ChestXray')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'datasets', 'chest_xray')
BBOX_CSV = os.path.join(DATA_DIR, 'BBox_List_2017.csv')
ENTRY_CSV = os.path.join(DATA_DIR, 'Data_Entry_2017.csv')

def setup_dirs():
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

def load_image_paths():
    print("Scanning for images...")
    paths = {}
    for img_path in glob.glob(os.path.join(DATA_DIR, 'images_*', 'images', '*.png')):
        paths[os.path.basename(img_path)] = img_path
    print(f"Found {len(paths)} images.")
    return paths

def load_bbox_data(image_paths):
    print("Loading BBox data...")
    # Map image_idx -> list of [class, x, y, w, h] (original)
    bboxes = {}
    classes = set()
    
    with open(BBOX_CSV, 'r') as f:
        reader = csv.reader(f)
        next(reader) # Skip header
        for row in reader:
            if not row: continue
            img_idx = row[0]
            label = row[1]
            x, y, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
            
            if img_idx not in image_paths:
                continue
                
            if img_idx not in bboxes:
                bboxes[img_idx] = []
            bboxes[img_idx].append({'label': label, 'bbox': [x, y, w, h]})
            classes.add(label)
    
    sorted_classes = sorted(list(classes))
    class_to_id = {cls: i for i, cls in enumerate(sorted_classes)}
    print(f"Classes found: {class_to_id}")
    return bboxes, sorted_classes, class_to_id

import struct

def get_png_size(file_path):
    """
    Reads the width and height of a PNG file from its header.
    """
    with open(file_path, 'rb') as f:
        data = f.read(26)
        # Check PNG signature
        if data[:8] != b'\x89PNG\r\n\x1a\n':
            return None
        # Check IHDR chunk type
        if data[12:16] != b'IHDR':
            return None
        # Read Width and Height (Big-endian)
        w, h = struct.unpack('>LL', data[16:24])
        return w, h

def load_image_dims(image_paths, bbox_images):
    print("Reading actual image dimensions from disk...")
    dims = {}
    # We only need dims for images with bboxes to normalize them.
    # We can check negative images too if needed, but for negatives we create empty files so size doesn't matter for normalization.
    
    # Process all bbox images
    for img_idx in bbox_images:
        if img_idx in image_paths:
            path = image_paths[img_idx]
            size = get_png_size(path)
            if size:
                dims[img_idx] = size
            else:
                print(f"Warning: Could not read size for {img_idx}")
                
    # Also process some negatives if we want to ensure we have a universe of valid images, 
    # but strictly we only need dims for *normalization*.
    # HOWEVER, we need dims to know which images exist/are valid?
    # get_negatives uses 'if img in dims'. So we should populate dims for ALL candidates 
    # OR change get_negatives logic.
    # To save time (reading 100k files might be slow), let's change logic:
    # We only strictly need dims for POSITIVES.
    # negative images can be assumed valid if they exist.
    
    return dims

def get_negatives(all_image_paths, bbox_images, dims, num_negatives=1000):
    # Select images that are in all_image_paths but NOT in bbox_images
    candidates = [img for img in all_image_paths if img not in bbox_images and img in dims]
    random.seed(42)
    selection = random.sample(candidates, min(len(candidates), num_negatives))
    print(f"Selected {len(selection)} negative images.")
    return selection

def convert_to_yolo(bbox, img_w, img_h):
    # bbox: x, y, w, h (top-left based)
    # YOLO: x_center, y_center, w, h (normalized)
    x, y, w, h = bbox
    
    # Clip to image boundaries
    x = max(0, x)
    y = max(0, y)
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    
    center_x = (x + w / 2) / img_w
    center_y = (y + h / 2) / img_h
    norm_w = w / img_w
    norm_h = h / img_h
    
    return center_x, center_y, norm_w, norm_h

def write_data(split, images, image_paths, bboxes, dims, class_to_id):
    print(f"Writing {split} data...")
    for img_idx in images:
        # Symlink Image
        src_path = image_paths[img_idx]
        dst_img_path = os.path.join(OUTPUT_DIR, 'images', split, img_idx)
        if os.path.exists(dst_img_path):
            os.remove(dst_img_path)
        os.symlink(src_path, dst_img_path)
        
        # Write Label
        label_path = os.path.join(OUTPUT_DIR, 'labels', split, os.path.splitext(img_idx)[0] + '.txt')
        with open(label_path, 'w') as f:
            if img_idx in bboxes:
                # Positive sample
                img_w, img_h = dims[img_idx]
                for item in bboxes[img_idx]:
                    cls_id = class_to_id[item['label']]
                    cx, cy, nw, nh = convert_to_yolo(item['bbox'], img_w, img_h)
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
            else:
                # Negative sample: empty file
                pass

def create_yaml(classes):
    yaml_content = f"""
path: {OUTPUT_DIR}
train: images/train
val: images/val
test: images/test

names:
"""
    for i, cls in enumerate(classes):
        yaml_content += f"  {i}: {cls}\n"
        
    with open(os.path.join(OUTPUT_DIR, 'data.yaml'), 'w') as f:
        f.write(yaml_content)
    print("Created data.yaml")


def main():
    setup_dirs()
    image_paths = load_image_paths()
    bboxes, classes, class_to_id = load_bbox_data(image_paths)
    
    # Only load dimensions for images that have bounding boxes
    dims = load_image_dims(image_paths, bboxes.keys())
    
    # Filter bboxes where we don't have dims (should be very few)
    valid_bbox_images = [img for img in bboxes if img in dims]
    print(f"Images with boxes and valid dimensions: {len(valid_bbox_images)}")
    
    # Get negatives using existence check instead of dims check
    # Check if we have random choice pool
    all_imgs = list(image_paths.keys())
    bbox_set = set(bboxes.keys())
    candidates = [img for img in all_imgs if img not in bbox_set]
    
    random.seed(42)
    negatives = random.sample(candidates, min(len(candidates), 500))
    print(f"Selected {len(negatives)} negative images.")
    
    all_images = valid_bbox_images + negatives
    random.shuffle(all_images)
    
    # Split
    total = len(all_images)
    train_end = int(total * 0.7)
    val_end = int(total * 0.85)
    
    train_imgs = all_images[:train_end]
    val_imgs = all_images[train_end:val_end]
    test_imgs = all_images[val_end:]
    
    write_data('train', train_imgs, image_paths, bboxes, dims, class_to_id)
    write_data('val', val_imgs, image_paths, bboxes, dims, class_to_id)
    write_data('test', test_imgs, image_paths, bboxes, dims, class_to_id)
    
    create_yaml(classes)
    print("Data preparation complete.")

if __name__ == '__main__':
    main()
