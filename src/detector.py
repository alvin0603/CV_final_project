import os
import sys
import glob
import torch
import cv2
import numpy as np
import supervision as sv
from torchvision.ops import box_convert

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
lib_path = os.path.join(project_root, "groundingdino_lib")
if lib_path not in sys.path:
    sys.path.append(lib_path)

try:
    from groundingdino.util.inference import load_model, load_image, predict
except ImportError:
    sys.exit(1)

class GroundingDINODetector:
    def __init__(self, config_path, weights_dir):
        weight_files = glob.glob(os.path.join(weights_dir, "*.pth"))
        if not weight_files:
            raise FileNotFoundError()
        
        weights_path = weight_files[0]
        self.model = load_model(config_path, weights_path)

    def run_inference(self, image_path, prompt, box_thresh, text_thresh):
        image_source, image = load_image(image_path)
        
        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=prompt,
            box_threshold=box_thresh,
            text_threshold=text_thresh
        )
        
        h, w, _ = image_source.shape
        
        padding = 100
        
        padded_image = cv2.copyMakeBorder(
            image_source, 
            padding, padding, padding, padding, 
            cv2.BORDER_CONSTANT, 
            value=(255, 255, 255)
        )
        
        boxes_scaled = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes_scaled, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        
        xyxy[:, [0, 2]] += padding
        xyxy[:, [1, 3]] += padding
        
        detections = sv.Detections(xyxy=xyxy)
        labels = [
            f"{phrase} {logit:.2f}"
            for phrase, logit
            in zip(phrases, logits)
        ]

        bbox_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
        label_annotator = sv.LabelAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            text_color=sv.Color.BLACK, 
            text_scale=0.4,
            text_thickness=1
        )
        
        annotated_frame = cv2.cvtColor(padded_image, cv2.COLOR_RGB2BGR)
        
        annotated_frame = bbox_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        
        max_score = 0.0
        if logits.numel() > 0:
            max_score = logits.max().item()

        return annotated_frame, len(boxes), max_score