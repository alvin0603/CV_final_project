import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
MODEL_CONFIG_PATH = os.path.join(PROJECT_ROOT, "groundingdino_lib", "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")

SAMPLE_NUM_PER_CLASS = 1e18
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

IOU_THRESHOLD = 0.5
CHESTXRAY_BBOX_CSV = os.path.join(PROJECT_ROOT, "downloads", "nih", "BBox_List_2017.csv")
BBOX_ACTIVE_DATASETS = ["medical"]

PROMPT_LEVEL = 1

MVTEC_L1 = {
    "bottle": "bottle",
    "cable": "cable",
    "capsule": "capsule",
    "carpet": "carpet",
    "grid": "grid",
    "hazelnut": "hazelnut",
    "leather": "leather",
    "metal_nut": "metal nut",
    "pill": "pill",
    "screw": "screw",
    "tile": "tile",
    "toothbrush": "toothbrush",
    "transistor": "transistor",
    "wood": "wood",
    "zipper": "zipper"
}

MEDICAL_L1 = {
    "Pneumonia": "pneumonia",
    "Nodule": "nodule",
    "Effusion": "pleural effusion",
    "Infiltration": "infiltration"
}

PATHOLOGY_L1 = {
    "cell": "tumor"
}

MVTEC_L2 = {
    "bottle": "broken bottle . glass crack . contamination . shattered glass .",
    "cable": "cut cable . bent wire . missing cable . twisted wire .",
    "capsule": "dented capsule . scratch . split . deformation .",
    "carpet": "carpet hole . cut . color stain . thread defect .",
    "grid": "bent grid . broken metal . distorted grid . metal debris .",
    "hazelnut": "hazelnut crack . hole . cut . print defect .",
    "leather": "leather cut . poke . bruise . fold . color stain .",
    "metal_nut": "metal nut scratch . bent . flip . color defect .",
    "pill": "pill scratch . crack . contamination . color defect . pill defect .",
    "screw": "screw thread defect . scratch . manipulated front .",
    "tile": "tile crack . glue . gray stroke . oil stain . rough surface .",
    "toothbrush": "defective bristles . missing bristles . bent stem .",
    "transistor": "bent lead . damaged casing . cut lead . misplaced .",
    "wood": "wood hole . scratch . liquid stain . knot . color spot .",
    "zipper": "broken zipper . fabric defect . split teeth . rough zipper ."
}

MEDICAL_L2 = {
    "Pneumonia": "lung opacity . hazy area . pulmonary consolidation . white patch .",
    "Nodule": "small round opacity . solitary pulmonary nodule . white spot . circular lesion .",
    "Effusion": "fluid accumulation . blunted costophrenic angle . white area at lung base .",
    "Infiltration": "ill-defined opacity . patchy shadow . lung texture anomaly . diffuse whiteness ."
}

PATHOLOGY_L2 = {
    "cell": "tumor cells . cancer metastasis . hyperchromatic nuclei . dense cell cluster . purple mass ."
}

MVTEC_L3 = {
    "bottle": "A glass bottle with visible structural damage and cracks on the surface .",
    "cable": "A wire cable that has been physically cut or bent sharply .",
    "capsule": "A pharmaceutical capsule showing signs of denting and crush damage .",
    "carpet": "A fabric carpet surface containing holes and color stains .",
    "grid": "A metal grid structure with bent wires and broken segments .",
    "hazelnut": "A hazelnut shell with visible cracks and holes drilled by insects .",
    "leather": "A leather texture surface showing cuts, bruises, and fold marks .",
    "metal_nut": "A metal nut with surface scratches and physical deformation .",
    "pill": "A medical pill with surface contamination and structural cracks .",
    "screw": "A metal screw with damaged threads and scratches on the head .",
    "tile": "A ceramic tile surface with visible glue spills and gray strokes .",
    "toothbrush": "A plastic toothbrush with missing and bent bristles .",
    "transistor": "An electronic transistor with damaged bent leads and casing .",
    "wood": "A wooden surface with liquid stains, scratches, and knot holes .",
    "zipper": "A clothing zipper with broken teeth and fabric tears ."
}

MEDICAL_L3 = {
    "Pneumonia": "A chest X-ray showing area of airspace consolidation indicative of pneumonia infection .",
    "Nodule": "A distinct small round opacity representing a solitary pulmonary nodule .",
    "Effusion": "A chest radiograph with blunted costophrenic angles due to pleural fluid accumulation .",
    "Infiltration": "An area of ill-defined patchy opacities suggesting pulmonary infiltration ."
}

PATHOLOGY_L3 = {
    "cell": "A histopathology slide showing a dense cluster of malignant tumor cells with dark hyperchromatic nuclei ."
}

if PROMPT_LEVEL == 1:
    SELECTED_MVTEC = MVTEC_L1
    SELECTED_MEDICAL = MEDICAL_L1
    SELECTED_PATHOLOGY = PATHOLOGY_L1
    MVTEC_DEFAULT = "defect"
    MEDICAL_DEFAULT = "disease"
    PATHOLOGY_DEFAULT = "cancer"
elif PROMPT_LEVEL == 2:
    SELECTED_MVTEC = MVTEC_L2
    SELECTED_MEDICAL = MEDICAL_L2
    SELECTED_PATHOLOGY = PATHOLOGY_L2
    MVTEC_DEFAULT = "defect . anomaly . damage ."
    MEDICAL_DEFAULT = "disease . abnormality ."
    PATHOLOGY_DEFAULT = "cell anomaly . cancer ."
else:
    SELECTED_MVTEC = MVTEC_L3
    SELECTED_MEDICAL = MEDICAL_L3
    SELECTED_PATHOLOGY = PATHOLOGY_L3
    MVTEC_DEFAULT = "An anomalous object with manufacturing defects ."
    MEDICAL_DEFAULT = "A chest X-ray showing radiological abnormalities ."
    PATHOLOGY_DEFAULT = "A microscopy slide showing malignant cancer tissue ."

DATASET_CONFIGS = {
    "MVTec": {
        "root": os.path.join(PROJECT_ROOT, "data", "MVTec"),
        "output": os.path.join(PROJECT_ROOT, "outputs", f"MVTec_Level{PROMPT_LEVEL}"),
        "categories": ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"],
        "prompts": SELECTED_MVTEC,
        "default_prompt": MVTEC_DEFAULT,
        "normal_folder": "good"
    },
    "medical": {
        "root": os.path.join(PROJECT_ROOT, "data", "medical"),
        "output": os.path.join(PROJECT_ROOT, "outputs", f"medical_Level{PROMPT_LEVEL}"),
        "categories": ["Pneumonia", "Nodule", "Effusion", "Infiltration"],
        "prompts": SELECTED_MEDICAL,
        "default_prompt": MEDICAL_DEFAULT,
        "normal_folder": "good"
    },
    "Pathology": {
        "root": os.path.join(PROJECT_ROOT, "data", "Pathology"),
        "output": os.path.join(PROJECT_ROOT, "outputs", f"Pathology_Level{PROMPT_LEVEL}"),
        "categories": ["cell"],
        "prompts": SELECTED_PATHOLOGY,
        "default_prompt": PATHOLOGY_DEFAULT,
        "normal_folder": "normal"
    }
}

ACTIVE_DATASETS = ["medical"]
