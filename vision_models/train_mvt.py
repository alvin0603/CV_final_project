import os
from ultralytics import YOLO

def main():

    # ---------------------------
    # 訓練參數設定（你可自由改）
    # ---------------------------
    model_name = "yolov8m-seg.pt"     # 初始模型
    data_yaml = "data.yaml"           # 你的 dataset config
    epochs = 100
    imgsz = 640
    batch = 8
    workers = 4

    # 如果要 resume，填入你的 last.pt
    resume_model = None  # "runs/segment/train/weights/last.pt"

    # ---------------------------
    # 開始訓練
    # ---------------------------
    if resume_model:
        print(f"Resuming training from: {resume_model}")
        model = YOLO(resume_model)
        model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            workers=workers,
            resume=True
        )
    else:
        print(f"Starting segmentation training with: {model_name}")
        model = YOLO(model_name)
        model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            workers=workers,
            save=True,
            project="runs/segment",
            name="mvtec_seg",
        )

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
