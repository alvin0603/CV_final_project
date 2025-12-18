from ultralytics import YOLO
import os

def main():
    # Initialize YOLOv8m model
    model = YOLO('yolov8m.pt')  # load a pretrained model (recommended for training)

    # Resolve paths relative to script
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
    DATA_YAML = os.path.join(PROJECT_ROOT, 'datasets', 'chest_xray', 'data.yaml')

    # Train the model
    # User requested mild augmentation suitable for Chest X-ray
    model.train(
        data=DATA_YAML,
        epochs=50,             # Reasonable default, user can stop early
        imgsz=640,
        batch=16,
        project=os.path.join(PROJECT_ROOT, 'yolo_result'),
        name='chestXray_v8m',
        exist_ok=True,
        
        # Augmentation hyperparameters (Mild)
        hsv_h=0.015,  # hue fractional
        hsv_s=0.7,    # saturation fractional
        hsv_v=0.4,    # value fractional
        degrees=10.0, # image rotation (+/- deg)
        translate=0.1,# image translation (+/- fraction)
        scale=0.1,    # image scale (+/- gain)
        shear=0.0,    # image shear (+/- deg)
        perspective=0.0, # image perspective (+/- fraction), range 0-0.001
        flipud=0.0,   # image flip up-down (not suitable for upright X-rays usually)
        fliplr=0.5,   # image flip left-right (suitable for symmetry)
        mosaic=0.0,   # mosaic (disabled for "mild" / preserving structure)
        mixup=0.0,    # mixup (disabled)
        copy_paste=0.0, # copy-paste (disabled)
    )

if __name__ == '__main__':
    main()
