import os
import torch
from ultralytics import YOLO

def main():

    # -------------------------------
    # 1. 選 GPU or CPU
    # -------------------------------
    if torch.cuda.is_available():
        device = 0
        print("使用 GPU 訓練")
    else:
        device = "cpu"
        print("找不到 GPU，改用 CPU")

    # -------------------------------
    # 2. 載入模型（可改 yolov8n/s/m/l/x）
    # -------------------------------
    model = YOLO("yolov8m.pt")  # 建議先用 s 模型

    # -------------------------------
    # 3. 訓練設定
    # -------------------------------
    results = model.train(
        data="/content/cv_final_yolo/detect.yaml",       # 你的 detect.yaml
        epochs=100,                # 訓練 epoch，可增加到 100
        imgsz=1024,               # ChestXray 是 1024x1024
        batch=8,                  # 如果 GPU 小可改成 batch=4
        device=device,            # GPU 編號 / CPU
        workers=4,
        name="cxr_detect_yolov8m",  # 實驗名稱（會出現在 runs/detect/）
        optimizer="Adam",         # 也可換 SGD
        lr0=1e-3,                 # 初始 learning rate
        patience=20,              # early stop
        pretrained=True           # 使用預訓練權重
    )

    # -------------------------------
    # 4. 評估（val split）
    # -------------------------------
    print("\n開始驗證 mAP...")
    metrics = model.val()
    print(metrics)

    # -------------------------------
    # 5. 對幾張圖片做推論（可注解掉）
    # -------------------------------
    sample_folder = "/content/cv_final_yolo/yolo_detect_dataset/images/val"
    print(f"\n推論 val 裡的樣本：{sample_folder}")

    predictions = model.predict(sample_folder, save=True, imgsz=1024)
    print("推論完成，結果已儲存在 runs/detect/predict/")

    print("\n訓練完成！最佳模型在：runs/detect/cxr_detect_yolov8s/weights/best.pt")

if __name__ == "__main__":
    main()
