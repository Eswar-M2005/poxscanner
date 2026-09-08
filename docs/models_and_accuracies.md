# Documentation: Evaluated Models and Diagnostic Accuracies

This document provides a comprehensive technical overview of **all deep learning models and architectures** evaluated in the Pox Scanner project, their design specifications, and their rigorous empirical accuracies on the holdout test dataset ($N = 270$ images across 12 isolated patient groups).

---

## 1. Executive Summary & Test Partition

All visual, tabular, and multimodal models were trained and benchmarked against the standardized holdout split manifests generated in [`data/splits/`](file:///Users/eswar.mandapati/Downloads/poxscanner/data/splits):

* **Train Partition (`train.csv`)**: 6,374 images (2,502 Monkeypox, 3,872 Non-Monkeypox) across 103 patient groups.
* **Validation Partition (`val.csv`)**: 272 images (52 Monkeypox, 220 Non-Monkeypox) across 12 patient groups.
* **Holdout Test Partition (`test.csv`)**: 270 images (52 Monkeypox, 218 Non-Monkeypox) across 12 patient groups.

> [!IMPORTANT]
> **Zero Data Leakage**: All splits enforce group-level isolation and near-duplicate perceptual hashing ($d_{\text{Hamming}} \le 5$) so that photos of the same patient or lesion are strictly kept within a single partition.

---

## 2. All Models Evaluated

The project investigated two layers of models:
1. **Computer Vision Backbones**: 4 pretrained convolutional architectures evaluated to identify the optimal visual feature extractor.
2. **Multimodal Diagnostic Systems**: 3 distinct operational systems combining visual imagery and tabular clinical symptoms (`Data.csv`).

```
                              ┌─────────────────────────────────────────────────┐
                              │               All Evaluated Models              │
                              └───────────────────────┬─────────────────────────┘
                                                      │
                       ┌──────────────────────────────┴──────────────────────────────┐
                       ▼                                                             ▼
         [ 1. Image-Only Backbones ]                                   [ 2. Multimodal Diagnostic Systems ]
         ├─ ResNet-50 (Selected)                                       ├─ Model A: Image-Only (ResNet-50)
         ├─ DenseNet-121                                               ├─ Model B: Clinical-Only (MLP)
         ├─ EfficientNet-B0                                            └─ Model C: Multimodal Fusion Network
         └─ MobileNet-V3-Small
```

---

## 3. Computer Vision Backbones Benchmark

To select the strongest vision backbone for lesion feature extraction, four architectures were trained with identical hyperparameter protocols (`AdamW`, weighted cross-entropy, data augmentation, batch size 16, 20 epochs):

| Architecture | Model Family | Parameters | Test Accuracy | Mpox Recall / Sens. | Specificity | Mpox F1 Score | ROC-AUC | Inference Latency (p50) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-50** *(Selected)* | Deep Residual Network | 25.6M | **86.67%** | **75.00%** | **89.45%** | **68.42%** | **91.97%** | 5.78 ms |
| **DenseNet-121** | Densely Connected Network | 8.0M | 82.22% | **75.00%** | 83.95% | 61.90% | 85.97% | 6.89 ms |
| **EfficientNet-B0** | Compound Scaling CNN | 5.3M | 81.48% | 59.62% | 86.70% | 55.36% | 86.53% | 50.97 ms |
| **MobileNet-V3-Small** | Mobile / Edge CNN | 2.5M | 81.85% | 44.23% | 90.83% | 48.42% | 81.69% | **2.04 ms** |

### Key Findings from Backbone Evaluation:
1. **ResNet-50** achieved the highest overall test accuracy (**86.67%**), best ROC-AUC (**91.97%**), and lowest generalization gap between validation and test sets (0.11 F1 gap).
2. **DenseNet-121** matched ResNet-50 in sensitivity (75.00%), but suffered more false positives, reducing specificity to 83.95%.
3. **MobileNet-V3-Small** provided the fastest CPU inference (2.04 ms), but showed lower sensitivity on lesion detection (44.23%).
4. **ResNet-50 was selected** as the foundational image backbone (`ml/artifacts/checkpoints/selected.pt`).

---

## 4. Multimodal Diagnostic Systems Benchmark

Using the selected ResNet-50 backbone and the clinical patient intake data (`Data.csv`), three primary operational models were evaluated on the holdout test set:

### Model Definitions

* **Model A (Image-Only)**: ResNet-50 backbone with global average pooling and linear classification head.
* **Model B (Clinical-Only)**: 2-layer Multi-Layer Perceptron (MLP) trained on 28 non-collinear ANOVA-ranked patient symptoms.
* **Model C (Multimodal Fusion)**: Joint neural network combining ResNet-50 visual features (256-d) with Clinical MLP embeddings (64-d) into a 320-d late fusion head.

### Comprehensive Performance Comparison Matrix

| Evaluation Metric | Model A: Image-Only | Model B: Clinical-Only | Model C: Multimodal Fusion | Multimodal Gain vs Image Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Accuracy** | 86.67% | **95.19%** | **90.37%** | **+3.70%** |
| **Balanced Accuracy** | 82.22% | **87.50%** | **84.52%** | **+2.30%** |
| **Precision (Monkeypox)** | 62.90% | **100.00%** | **75.00%** | **+12.10%** |
| **Recall / Sensitivity** | **75.00%** | **75.00%** | **75.00%** | Preserved (0.00%) |
| **Specificity (Non-Monkeypox)**| 89.45% | **100.00%** | **94.04%** | **+4.59%** |
| **Macro F1 Score** | 79.99% | **91.41%** | **84.52%** | **+4.53%** |
| **Weighted F1 Score** | 87.09% | **94.91%** | **90.37%** | **+3.28%** |
| **ROC-AUC** | 91.97% | **97.13%** | **93.60%** | **+1.63%** |
| **False Positives ($n=218$)** | 23 | **0** | **13** | **-43.5% reduction** |
| **False Negatives ($n=52$)** | 13 | 13 | 13 | Maintained |

---

## 5. Confusion Matrices & Error Breakdown

Evaluated on the $N = 270$ holdout test partition:

### Model A: Image-Only Baseline
```
                     Predicted Non-Mpox    Predicted Mpox
Actual Non-Mpox:            195                  23   (TN: 195, FP: 23)
Actual Mpox:                 13                  39   (FN: 13,  TP: 39)
```
* **Accuracy**: $(195 + 39) / 270 = \mathbf{86.67\%}$
* *Limitation*: Visual similarities between atypical non-monkeypox lesions and true monkeypox lesions caused 23 false positives.

### Model B: Clinical-Only Baseline
```
                     Predicted Non-Mpox    Predicted Mpox
Actual Non-Mpox:            218                   0   (TN: 218, FP: 0)
Actual Mpox:                 13                  39   (FN: 13,  TP: 39)
```
* **Accuracy**: $(218 + 39) / 270 = \mathbf{95.19\%}$
* *Strength*: Hallmark clinical symptoms (mucosal ulcers, high fever, lymphadenopathy) provided zero false positives on this partition.

### Model C: Multimodal Fusion Network
```
                     Predicted Non-Mpox    Predicted Mpox
Actual Non-Mpox:            205                  13   (TN: 205, FP: 13)
Actual Mpox:                 13                  39   (FN: 13,  TP: 39)
```
* **Accuracy**: $(205 + 39) / 270 = \mathbf{90.37\%}$
* *Synergy*: Integrating clinical presentation reduced false positives from 23 down to 13, boosting precision from 62.90% to **75.00%** while maintaining 75.00% recall.

---

## 6. Dynamic Runtime Routing Architecture

The production application dynamically selects the appropriate model according to the inputs supplied by the clinician or researcher:

```
[ User Request ]
       │
       ├─► Has ONLY Clinical Symptoms  ──► [ Model B: Clinical MLP ]
       │                                     • Accuracy: 95.19%
       │                                     • Output: Symptom Attributions
       │
       ├─► Has ONLY Lesion Photograph  ──► [ Model A: ResNet-50 ]
       │                                     • Accuracy: 86.67%
       │                                     • Output: Grad-CAM Overlay
       │
       └─► Has BOTH Image & Symptoms   ──► [ Model C: Multimodal Fusion ]
                                             • Accuracy: 90.37%
                                             • Output: Grad-CAM + Symptom Attributions
```

---

## 7. Model Checkpoint & Metric Artifact References

| Model Name | Artifact Checkpoint | Test Metric File |
| :--- | :--- | :--- |
| **ResNet-50** (Image Baseline) | [`ml/artifacts/checkpoints/resnet50_best.pt`](file:///Users/eswar.mandapati/Downloads/poxscanner/ml/artifacts/checkpoints/resnet50_best.pt) | [`ml/artifacts/metrics/resnet50_test.json`](file:///Users/eswar.mandapati/Downloads/poxscanner/ml/artifacts/metrics/resnet50_test.json) |
| **DenseNet-121** (Candidate) | [`ml/artifacts/checkpoints/densenet121_best.pt`](file:///Users/eswar.mandapati/Downloads/poxscanner/ml/artifacts/checkpoints/densenet121_best.pt) | [`ml/artifacts/metrics/densenet121_test.json`](file:///Users/eswar.mandapati/Downloads/poxscanner/ml/artifacts/metrics/densenet121_test.json) |
| **EfficientNet-B0** (Candidate) | [`ml/artifacts/checkpoints/efficientnet_b0_best.pt`](file:///Users/eswar.mandapati/Downloads/poxscanner/ml/artifacts/checkpoints/efficientnet_b0_best.pt) | [`ml/artifacts/metrics/efficientnet_b0_test.json`](file:///Users/eswar.mandapati/Downloads/poxscanner/ml/artifacts/metrics/efficientnet_b0_test.json) |
| **MobileNet-V3-Small** (Candidate) | [`ml/artifacts/checkpoints/mobilenet_v3_small_best.pt`](file:///Users/eswar.mandapati/Downloads/poxscanner/ml/artifacts/checkpoints/mobilenet_v3_small_best.pt) | [`ml/artifacts/metrics/mobilenet_v3_small_test.json`](file:///Users/eswar.mandapati/Downloads/poxscanner/ml/artifacts/metrics/mobilenet_v3_small_test.json) |
| **Model B: Clinical-Only MLP** | [`ml/artifacts/checkpoints/clinical_only_best.pt`](file:///Users/eswar.mandapati/Downloads/poxscanner/ml/artifacts/checkpoints/clinical_only_best.pt) | [`ml/artifacts/metrics/model_comparison.json`](file:///Users/eswar.mandapati/Downloads/poxscanner/ml/artifacts/metrics/model_comparison.json) |
| **Model C: Multimodal Fusion** | [`ml/artifacts/checkpoints/multimodal_best.pt`](file:///Users/eswar.mandapati/Downloads/poxscanner/ml/artifacts/checkpoints/multimodal_best.pt) | [`ml/artifacts/metrics/model_comparison.json`](file:///Users/eswar.mandapati/Downloads/poxscanner/ml/artifacts/metrics/model_comparison.json) |
