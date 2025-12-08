import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
MODEL_CONFIG_PATH = os.path.join(PROJECT_ROOT, "groundingdino_lib", "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")

SAMPLE_NUM_PER_CLASS = 1e18
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

DATASET_CONFIGS = {
    "MVTec": {
        "root": os.path.join(PROJECT_ROOT, "data", "MVTec"),
        "output": os.path.join(PROJECT_ROOT, "outputs", "MVTec_Eval"),
        "categories": ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"],
        "prompts": {
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
        },
        "default_prompt": "defect . anomaly .",
        "normal_folder": "good"
    },
    "X-Ray": {
        "root": os.path.join(PROJECT_ROOT, "data", "medical"), 
        "output": os.path.join(PROJECT_ROOT, "outputs", "medical"),
        "categories": ["Pneumonia", "Nodule", "Effusion", "Infiltration"],
        "prompts": {
            "Pneumonia": "pneumonia",
            "Nodule": "nodule",
            "Effusion": "pleural effusion",
            "Infiltration": "infiltration"
        },
        "default_prompt": "disease",
        "normal_folder": "good"
    },
    "Pathology": {
        "root": os.path.join(PROJECT_ROOT, "data", "Pathology"),
        "output": os.path.join(PROJECT_ROOT, "outputs", "Pathology_Eval"),
        "categories": ["cell"],
        "prompts": {
            "cell": "cancer . tumor . metastasis . malignant . carcinoma ."
        },
        "default_prompt": "cell anomaly . cancer .",
        "normal_folder": "normal"
    }
}

ACTIVE_DATASETS = ["X-Ray"]