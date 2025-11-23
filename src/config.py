import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
MODEL_CONFIG_PATH = os.path.join(PROJECT_ROOT, "groundingdino_lib", "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")

SAMPLE_NUM_PER_CLASS = 100
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

MVTEC_PROMPTS = {
    "bottle": "bottle . broken bottle . glass defect .",
    "cable": "cable . cut cable . bent wire .",
    "capsule": "capsule . dented capsule . scratch .",
    "carpet": "carpet . hole . cut . color stain .",
    "grid": "grid . bent grid . broken metal .",
    "hazelnut": "hazelnut . hole . crack .",
    "leather": "leather . cut . poke . bruise .",
    "metal_nut": "metal nut . scratch . bent .",
    "pill": "pill . scratch . contamination .",
    "screw": "screw . thread defect . scratch .",
    "tile": "tile . crack . glue . gray stroke .",
    "toothbrush": "toothbrush . defective bristles .",
    "transistor": "transistor . bent lead . damaged .",
    "wood": "wood . hole . scratch . liquid .",
    "zipper": "zipper . broken zipper . fabric defect ."
}

CHEST_XRAY_PROMPTS = {
    "lung": "pneumonia . lung opacity . effusion . infiltration . mass ."
}

PATHOLOGY_PROMPTS = {
    "cell": "cancer . tumor . metastasis . malignant . carcinoma ."
}

DATASET_CONFIGS = {
    "MVTec": {
        "root": os.path.join(PROJECT_ROOT, "data", "MVTec"),
        "output": os.path.join(PROJECT_ROOT, "outputs", "MVTec_Eval"),
        "categories": list(MVTEC_PROMPTS.keys()),
        "prompts": MVTEC_PROMPTS,
        "default_prompt": "defect . anomaly .",
        "normal_folder": "good"
    },
    "ChestXray": {
        "root": os.path.join(PROJECT_ROOT, "data", "ChestXray"),
        "output": os.path.join(PROJECT_ROOT, "outputs", "ChestXray_Eval"),
        "categories": list(CHEST_XRAY_PROMPTS.keys()),
        "prompts": CHEST_XRAY_PROMPTS,
        "default_prompt": "medical anomaly . disease .",
        "normal_folder": "normal"
    },
    "Pathology": {
        "root": os.path.join(PROJECT_ROOT, "data", "Pathology"),
        "output": os.path.join(PROJECT_ROOT, "outputs", "Pathology_Eval"),
        "categories": list(PATHOLOGY_PROMPTS.keys()),
        "prompts": PATHOLOGY_PROMPTS,
        "default_prompt": "cell anomaly . cancer .",
        "normal_folder": "normal"
    }
}

ACTIVE_DATASETS = ["Pathology"]