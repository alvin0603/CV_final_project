
# Slide 1: Title

## **Analyzing and Improving the Generalization Capabilities of Grounding DINO on Unseen Domains**

### Can Open-Set Detectors Generalize to Specialized Domains?

**Group 8:**
- 黃皓群 (B123245009)
- 周霖 (B123245028)
- 陳圖億 (B123040002)

---

# Slide 2: Problem Definition

## **Problem Definition & Motivation**

### The Shift: Closed-Set → Open-Set Detection
- **Traditional models** (DETR, YOLO): Can only detect predefined categories
- **Open-set models** (Grounding DINO): Detect arbitrary objects via text descriptions

### Grounding DINO's Achievements
- 52.5 AP on COCO zero-shot benchmark
- State-of-the-art on ODinW
- Combines DINO Transformer with grounded pre-training

### Research Gap (Our Motivation)
> **"Despite strong results on COCO/LVIS, the model's zero-shot transferability to specialized domains with large domain gaps remains largely unexplored."**

- Industrial defect detection (MVTec)
- Medical imaging (ChestX-ray)
- Pathology slides (PatchCamelyon)

### Project Goal
1. **Evaluate** zero-shot performance boundaries on unseen domains
2. **Quantify** the effectiveness of few-shot fine-tuning for domain adaptation

---

# Slide 3: Datasets

## **Dataset Introduction**

| Dataset | Type | Categories | Description |
|---------|------|------------|-------------|
| **MVTec-AD** | Industrial Defects | 15 | Manufacturing benchmark |
| **ChestX-ray** | Medical Imaging | 4 | Lung X-ray lesion detection |
| **PatchCamelyon** | Pathology Slides | 1 | Tumor cell recognition |

### Evaluation Methods
- **Image-Level**: Classify whether image contains anomaly
- **BBox-Level**: Localize anomaly position (mAP, IoU)

**Content to Include:**
- Sample images from each dataset
- Dataset statistics summary table

---

# Slide 4: Methodology - GroundingDINO

## **GroundingDINO Overview**

### Background
- **DETR** (2020): End-to-End Object Detection with Transformers
- **DINO** (2022): Self-Distillation with NO Labels
- **GroundingDINO** (2023): Open-Set Object Detection

### Key Features
- Combines **Vision Transformer** with **BERT** for vision-language
- Supports **arbitrary text descriptions** as detection targets
- Detects novel classes without retraining

### Image-Level Evaluation Method
```
Input: Image + Text description (e.g., "defect")
     ↓
GroundingDINO Detection
     ↓
Take max(confidence) as anomaly score
     ↓
Compute AUC / AP
```

**Content to Include:**
- GroundingDINO architecture diagram (official)
- Model evolution timeline: DETR → DINO → GroundingDINO

---

# Slide 5: Few-Shot Tuning Methods

## **Few-Shot Tuning Methods**

### Training Settings
- **K-Shot**: 32 (32 normal + 32 anomaly per category)
- **Epochs**: 100
- **Loss**: Margin-based ranking loss

---

# Slide 5a: Prompt Tuning

## **Prompt Tuning**

### Principle
- Freeze entire model, **only train additional prompt embeddings**
- These embeddings are added to original text embeddings

### Implementation
```
Original text embedding: [CLS] defect [SEP]
                          ↓
Add learnable prompts: [P1][P2]...[P8] + [CLS] defect [SEP]
                          ↓
GroundingDINO forward
```

### Parameters
- ~**2K parameters** (8 prompt tokens × 256 dim)

### Results
- **Ineffective**: Some categories degraded
- Reason: GroundingDINO architecture cannot effectively utilize enhanced embeddings

---

# Slide 5b: LoRA

## **LoRA (Low-Rank Adaptation)**

### Principle
- Add low-rank update to original weight matrix: **W' = W + BA**
- B ∈ R^(d×r), A ∈ R^(r×k), where r << min(d, k)

### Implementation
```
Original Linear layer: y = Wx
                       ↓
LoRA modification:     y = Wx + BAx
                       ↓
Only train A and B matrices
```

### Injection Location
- Query/Key/Value projection layers (~20 layers)

### Parameters
- ~**80K parameters** (rank=4)

### Results
- **Most stable and effective**: Improvement on all datasets

---

# Slide 5c: LoRA + Adapter

## **LoRA + Adapter**

### Principle
- Combine LoRA and Adapter methods
- Adapter: Add bottleneck MLP after layer output

### Implementation
```
LoRA output: y = Wx + BAx
             ↓
Adapter:     y' = y + MLP(LayerNorm(y))
             ↓
MLP structure: Down(256→64) → GELU → Up(64→256)
```

### Parameters
- ~**150K parameters** (LoRA + Adapter)

### Results
- Better than pure LoRA on some datasets (X-ray, PCAM)
- Less stable on MVTec

---

# Slide 5d: Method Comparison

## **Visual Comparison of Three Methods**

**Content to Include:**
Architecture comparison diagram:

```
┌─────────────────────────────────────────────────────────────┐
│  Prompt Tuning    │     LoRA          │   LoRA + Adapter   │
├───────────────────┼───────────────────┼────────────────────┤
│                   │                   │                    │
│  [Learnable       │   W + BA          │   W + BA           │
│   Prompts]        │                   │      ↓             │
│      ↓            │                   │   Adapter          │
│  Text Encoder     │   Linear Layer    │                    │
│                   │                   │                    │
├───────────────────┼───────────────────┼────────────────────┤
│  Params: ~2K      │   Params: ~80K    │   Params: ~150K    │
│  Effect: Poor     │   Effect: Good    │   Effect: Good     │
└─────────────────────────────────────────────────────────────┘
```

---

# Slide 6: ResNet Baseline Results

## **Supervised Learning Baseline (ResNet)**

> **Conclusion: Supervised learning performs excellently with sufficient data**

### MVTec-AD (ResNet)

| Category | AUC | AP |
|----------|-----|-----|
| bottle | **1.00** | 1.00 |
| cable | **0.99** | 0.99 |
| capsule | **1.00** | 1.00 |
| carpet | **1.00** | 1.00 |
| leather | **1.00** | 1.00 |
| ... | ... | ... |
| **Average** | **~0.99** | **~0.99** |

### ChestX-ray (ResNet)

| Category | AUC | AP |
|----------|-----|-----|
| Pneumonia | 0.75 | 0.31 |
| Nodule | 0.72 | 0.58 |
| Effusion | 0.80 | 0.82 |
| Infiltration | 0.73 | 0.81 |
| **Average** | **0.75** | **0.63** |

**Key Points:**
- MVTec: Near perfect (AUC ~0.99)
- ChestX-ray: More challenging but still good performance
- Takeaway: With sufficient labeled data, supervised learning remains a strong baseline

---

# Slide 7: GroundingDINO Zero-Shot Results

## **Zero-Shot Performance**

### MVTec-AD

| Category | Zero-shot AUC | Observation |
|----------|---------------|-------------|
| leather | **0.92** | Best |
| tile | 0.76 | Good |
| bottle | 0.70 | Moderate |
| carpet | 0.12 | Very poor |
| metal_nut | 0.20 | Very poor |
| **Average** | **~0.49** | Near random |

### ChestX-ray

| Category | Zero-shot AUC |
|----------|---------------|
| Pneumonia | 0.56 |
| Nodule | 0.51 |
| Effusion | 0.55 |
| Infiltration | 0.49 |
| **Average** | **~0.53** | Near random |

### PatchCamelyon

| Category | Zero-shot AUC |
|----------|---------------|
| cell | **0.60** |

**Key Points:**
- Zero-shot performance generally near random (AUC ~0.5)
- Few categories perform well (leather 0.92)
- **Conclusion: Pure zero-shot is insufficient for anomaly detection**

---

# Slide 7b: Prompt Design Analysis

## **Does Complex Prompt Help? (Level 1 vs Level 2)**

### Prompt Examples

| Level | MVTec Example | X-ray Example |
|-------|---------------|---------------|
| **Level 1** (Simple) | "defect" | "pneumonia" |
| **Level 2** (Complex) | "broken bottle . glass crack . contamination ." | "lung opacity . consolidation . infiltrate ." |

### MVTec-AD: Level 1 vs Level 2 AUC

| Category | Level 1 | Level 2 | Difference |
|----------|---------|---------|------------|
| bottle | **0.70** | 0.44 | -26% |
| leather | **0.92** | 0.07 | -85% |
| tile | **0.76** | 0.66 | -10% |
| wood | **0.27** | 0.17 | -10% |
| **Average** | **~0.49** | ~0.41 | **-8%** |

### ChestX-ray: Level 1 vs Level 2 AUC

| Category | Level 1 | Level 2 | Difference |
|----------|---------|---------|------------|
| Pneumonia | **0.56** | 0.48 | -8% |
| Effusion | **0.55** | 0.36 | -19% |
| **Average** | **~0.53** | ~0.47 | **-6%** |

**Key Finding:**
> **Complex prompts do NOT improve performance — they actually degrade it.**

Possible reasons:
1. Over-specific descriptions may not match visual features
2. Multiple terms may cause confusion in cross-modal alignment
3. Simple, generic prompts allow model to use broader learned representations

---

# Slide 8: LoRA Improvement (KEY SLIDE)

## **LoRA Few-Shot Tuning Results**

### MVTec-AD: Zero-shot vs LoRA

| Category | Zero-shot | LoRA | **Improvement** |
|----------|-----------|------|-----------------|
| bottle | 0.70 | 0.73 | +3% |
| cable | 0.39 | **0.58** | **+19%** |
| capsule | 0.51 | **0.70** | **+19%** |
| carpet | 0.12 | **0.64** | **+52%** |
| leather | 0.92 | **0.94** | +2% |
| metal_nut | 0.20 | **0.54** | **+34%** |
| wood | 0.27 | **0.83** | **+56%** |
| **Average** | ~0.49 | **~0.58** | **+9%** |

### ChestX-ray: Zero-shot vs LoRA

| Category | Zero-shot | LoRA | **Improvement** |
|----------|-----------|------|-----------------|
| Pneumonia | 0.56 | **0.61** | +5% |
| Nodule | 0.51 | **0.52** | +1% |
| Effusion | 0.55 | **0.60** | +5% |
| Infiltration | 0.49 | **0.57** | **+8%** |
| **Average** | ~0.53 | **~0.57** | **+4%** |

### PatchCamelyon: Zero-shot vs LoRA

| Method | AUC | AP | **Improvement** |
|--------|-----|-----|-----------------|
| Zero-shot | 0.60 | 0.54 | - |
| **LoRA** | **0.80** | **0.78** | **+20%** |

**Key Points:**
- LoRA shows significant improvement on all datasets
- Largest improvements: carpet (+52%), wood (+56%), PatchCamelyon (+20%)
- Proves few-shot tuning is effective for GroundingDINO

---

# Slide 9: Method Comparison Summary

## **All Methods Comparison Summary**

### MVTec-AD Average AUC

| Method | Avg AUC | vs Zero-shot |
|--------|---------|--------------|
| Zero-shot | 0.49 | - |
| Prompt Tuning | 0.34 | -15% |
| **LoRA** | **0.58** | **+9%** |
| LoRA + Adapter | 0.52 | +3% |

### ChestX-ray Average AUC

| Method | Avg AUC | vs Zero-shot |
|--------|---------|--------------|
| Zero-shot | 0.53 | - |
| **LoRA** | **0.57** | **+4%** |
| LoRA + Adapter | **0.64** | **+11%** |

### PatchCamelyon AUC

| Method | AUC | vs Zero-shot |
|--------|-----|--------------|
| Zero-shot | 0.60 | - |
| Prompt Tuning | 0.46 | -14% |
| **LoRA** | **0.80** | **+20%** |
| **LoRA + Adapter** | **0.84** | **+24%** |

**Key Findings:**
1. **LoRA is the most stable and effective method**
2. Prompt Tuning is unstable, some degradation
3. LoRA + Adapter performs better on some datasets

---

# Slide 10: BBox-Level Challenges

## **BBox-Level Evaluation Challenges**

### Results Summary
- All methods achieved mAP close to **0** at BBox-level
- Even when boxes were detected, IoU was very low

### Root Cause Analysis
1. **Domain Gap**: GroundingDINO trained on natural images, medical/industrial images are very different
2. **Imprecise Prompts**: Simple "defect" cannot accurately describe anomaly location
3. **Anomaly Granularity**: Anomaly regions may be very small (cracks) or diffuse (infiltration)

### Recommendations
- Image-level evaluation is more suitable for open-vocabulary models
- BBox-level may require specialized fine-tuning or different model design

**Content to Include:**
- Brief text explanation only, no detailed data tables needed

---

# Slide 11: Conclusion & Future Work

## **Conclusion & Future Work**

### Conclusions
1. **Supervised learning is still king**: ResNet achieves AUC ~0.99 with sufficient data
2. **Zero-shot is insufficient**: GroundingDINO pure zero-shot is near random
3. **LoRA is effective**: Few-shot LoRA improves AUC by 4%~20%
4. **BBox remains challenging**: Open-vocabulary models struggle with precise localization

---

