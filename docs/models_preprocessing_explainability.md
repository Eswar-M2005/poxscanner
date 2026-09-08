# Technical Reference: Models, Data Preprocessing, and Explainability

This document provides a comprehensive technical reference for the models, data preprocessing pipelines, and dual-modality explainability methods implemented in the Pox Scanner multimodal classification system.

---

## 1. System Overview & Pipeline Architecture

Pox Scanner is an educational and research tool designed for binary skin lesion classification (**Monkeypox** vs. **Non-Monkeypox**). The system combines:
1. **Computer Vision**: Convolutional neural networks extracting spatial visual features from dermatological photographs.
2. **Tabular Clinical Data**: Multi-layer perceptrons encoding clinical symptom presentations from patient intake forms.
3. **Multimodal Late Fusion**: A unified neural classification head synthesizing visual and clinical representations.
4. **Dual Explainability**: Grad-CAM visual heatmaps paired with clinical symptom perturbation attributions.

```
                  ┌─────────────────────────────────────┐
                  │       Dermatological Image          │
                  └──────────────────┬──────────────────┘
                                     │ (224x224x3 Tensor)
                                     ▼
                  ┌─────────────────────────────────────┐
                  │    ResNet-50 Feature Backbone       │
                  │   (layer1-layer4 + Adaptive Pool)   │
                  └──────────────────┬──────────────────┘
                                     │ (256-d Image Vector)
                                     │
┌───────────────────────────────┐    │
│  Patient Clinical Symptoms    │    │
│       (Data.csv)              │    │
└──────────────┬────────────────┘    │
               │ (28-d Vector)       │
               ▼                     │
┌───────────────────────────────┐    │
│   Clinical Preprocessor       │    │
│  (Leakage Removal + Scaling)  │    │
└──────────────┬────────────────┘    │
               │                     │
               ▼                     │
┌───────────────────────────────┐    │
│         Clinical MLP          │    │
│    (2x Linear + BN + ReLU)    │    │
└──────────────┬────────────────┘    │
               │ (64-d Vector)       │
               └──────────┬──────────┘
                          │
                          ▼
        ┌───────────────────────────────────┐
        │       Late Fusion Module          │
        │ Concatenation (320-d)             │
        │ Linear(320->128) + LayerNorm      │
        │ Dropout(0.3) + Linear(128->2)     │
        └─────────────────┬─────────────────┘
                          │
                          ▼
            [ Monkeypox / Non-Monkeypox ]
            [ Probabilities & Attributions ]
```

### 1.1 Dynamic Modality Dispatching

The inference engine dynamically dispatches incoming prediction requests to the appropriate model based on provided user inputs:

| User Inputs Provided | Active Architecture | Model Checkpoint | Explainability Output |
| :--- | :--- | :--- | :--- |
| **Only Clinical Symptoms** (No Image) | **Model B**: Clinical-Only MLP | `clinical_only_best.pt` | Symptom Attribution (Ranked supporting & opposing factors) |
| **Only Lesion Photograph** (No Symptoms) | **Model A**: ResNet-50 (Image-Only) | `selected.pt` / `resnet50_best.pt` | Grad-CAM Spatial Heatmap Overlay |
| **Both Image & Clinical Symptoms** | **Model C**: Multimodal Fusion Network | `multimodal_best.pt` | **Dual Explainability**: Grad-CAM Overlay + Symptom Attribution |

---

## 2. Data Preprocessing Pipelines

### 2.1 Image Data Preprocessing & Augmentation

The image preprocessing pipeline is implemented in [`ml/src/transforms.py`](file:///Users/eswar.mandapati/Downloads/poxscanner/ml/src/transforms.py) and [`ml/src/predict.py`](file:///Users/eswar.mandapati/Downloads/poxscanner/ml/src/predict.py).

#### Training Pipeline (Augmentation & Regularization)
Images are transformed dynamically per training batch to enhance generalizability:
- **Resizing**: Scaled to $256 \times 256$ pixels using bilinear interpolation.
- **Random Cropping / Scaling**: Centered or random crop to standard model input dimensions ($224 \times 224$).
- **Geometric Invariance**:
  - `RandomHorizontalFlip(p=0.5)`
  - `RandomVerticalFlip(p=0.5)`
  - `RandomRotation(degrees=15)`
- **Photometric Invariance**:
  - `ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15)`
- **Tensor Conversion & Channel Normalization**:
  - Scaled to floating point $[0.0, 1.0]$.
  - Normalized using standard ImageNet distribution statistics:
    $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$

#### Validation & Inference Pipeline (Deterministic Evaluation)
For validation, testing, and production inference:
- **Resize**: Rescaled to $256 \times 256$.
- **Center Crop**: Uniform crop to $224 \times 224$ pixels.
- **Normalization**: Same ImageNet $\mu$ and $\sigma$ channel standardization without stochastic distortions.

---

### 2.2 Clinical Data Preprocessing Pipeline

Implemented in [`ml/src/clinical.py`](file:///Users/eswar.mandapati/Downloads/poxscanner/ml/src/clinical.py) (`ClinicalPreprocessor`).

#### Source Dataset Characteristics (`Data.csv`)
- **Dimensions**: 211 patient rows, 48 features (all `int64`).
- **Target Label**: `Status` ($1 = \text{Monkeypox}$, $0 = \text{Non-Monkeypox}$).
- **Distribution**: 186 positive cases, 25 negative cases (7.44:1 class imbalance).

#### Preprocessing Steps

1. **Identifier & Direct Target Exclusion**:
   - Column `ID` (opaque sequence numbers 1–211) and `Status` are immediately stripped.
2. **Clinical Leakage Prevention**:
   - Explicitly dropped columns:
     - `characteristic symptoms of Monkeypox`
     - `symptoms compatible with monkeypox`
   - *Rationale*: These columns represent post-hoc clinician summary assessments rather than raw observational symptoms; retaining them creates severe artificial target leakage.
3. **Collinearity and Variance Filtering**:
   - Pruned zero-variance features across patient groups.
   - Identified and pruned redundant collinear pairs (e.g., `perianal papules` and `inguinal adenopathy` having Pearson correlation $r = 1.0$).
4. **Univariate Feature Ranking**:
   - Computed on the training split using ANOVA F-value ranking:
     $$F = \frac{\text{Between-group variance}}{\text{Within-group variance}}$$
   - Top 28 informative symptoms are retained (e.g., `oral and genital ulcers`, `ulcerative lesions`, `fever`, `rash`, `swollen lymph nodes`, `muscle pain`, `pustules`, `chills`).
5. **Missing Value Imputation & Standardization**:
   - Missing symptoms default to $0$ (absence).
   - Zero-mean, unit-variance standard scaling:
     $$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}} + \epsilon}$$
   - Parameters saved to [`ml/artifacts/metrics/clinical_features.json`](file:///Users/eswar.mandapati/Downloads/poxscanner/ml/artifacts/metrics/clinical_features.json) to guarantee identical inference preprocessing.

---

### 2.3 Multimodal Dataset Assembly & Split Isolation

Implemented in [`ml/src/dataset.py`](file:///Users/eswar.mandapati/Downloads/poxscanner/ml/src/dataset.py) (`MultimodalLesionDataset`).

- **Group-Level Isolation**: The dataset partitions (`train.csv`, `val.csv`, `test.csv`) strictly segment data by patient identity:
  - `train.csv`: 6,374 images (103 patient groups)
  - `val.csv`: 272 images (12 patient groups)
  - `test.csv`: 270 images (12 patient groups)
- **Modality Synchronization**: Each sample returns:
  ```python
  {
      "image": torch.FloatTensor,    # [3, 224, 224]
      "clinical": torch.FloatTensor, # [28]
      "label": int,                  # 0 (non_monkeypox) or 1 (monkeypox)
      "group_id": str,               # Patient group identifier
      "path": str                    # Image path
  }
  ```
- Images belonging to the same patient subject reference the corresponding clinical feature vector without cross-contaminating validation or test partitions.

---

## 3. Models Used for Training

The project benchmarks three distinct model configurations to assess the individual and synergistic contributions of each modality.

### 3.1 Model A: Image-Only Baseline

- **Architecture**: Deep Residual Network (`ResNet-50`).
- **Backbone**: 50 layers with bottleneck residual blocks and shortcut connections.
- **Visual Projection**:
  - Global Average Pooling $\to$ Flatten $\to$ Linear classifier head ($2048 \to 2$).
- **Input**: $3 \times 224 \times 224$ normalized RGB image.
- **Output**: 2-class logits.

---

### 3.2 Model B: Clinical-Only Baseline

Implemented in [`ml/src/architectures/multimodal.py`](file:///Users/eswar.mandapati/Downloads/poxscanner/ml/src/architectures/multimodal.py) (`ClinicalOnlyClassifier`).

- **Architecture**: Multi-Layer Perceptron (`ClinicalMLP`).
- **Layers**:
  1. `Linear(in_features=28, out_features=128)`
  2. `BatchNorm1d(128)`
  3. `ReLU(inplace=True)`
  4. `Dropout(p=0.25)`
  5. `Linear(in_features=128, out_features=64)`
  6. `BatchNorm1d(64)`
  7. `ReLU(inplace=True)`
  8. `Linear(in_features=64, out_features=2)`
- **Input**: 28-dimensional continuous symptom vector.
- **Output**: 2-class logits.

---

### 3.3 Model C: Multimodal Fusion Network

Implemented in [`ml/src/architectures/multimodal.py`](file:///Users/eswar.mandapati/Downloads/poxscanner/ml/src/architectures/multimodal.py) (`MultimodalClassifier`).

Combines the visual feature extractor and the clinical encoder via late fusion:

1. **Image Feature Branch**:
   - `ResNet-50` backbone truncated prior to classification.
   - Global Adaptive Average Pooling ($2048 \times 1 \times 1$).
   - Linear projection: $\text{Linear}(2048 \to 256)$ with batch normalization and dropout.
   - Generates a **256-dimensional** visual representation $v_{\text{img}}$.

2. **Clinical Feature Branch**:
   - `ClinicalMLP` encoder mapping raw symptoms to a dense embedding.
   - Generates a **64-dimensional** clinical representation $v_{\text{clin}}$.

3. **Multimodal Late Fusion Head**:
   - Concatenation of modality vectors:
     $$v_{\text{fused}} = [v_{\text{img}} \,\|\, v_{\text{clin}}] \in \mathbb{R}^{320}$$
   - Dense Transformation:
     $$\text{Linear}(320 \to 128) \to \text{LayerNorm}(128) \to \text{ReLU} \to \text{Dropout}(p=0.3)$$
   - Output Projection:
     $$\text{Linear}(128 \to 2)$$
   - Produces unnormalized logits $[\ell_0, \ell_1]$ for Non-Monkeypox and Monkeypox.

---

## 4. Training Methodology & Optimization

Implemented in [`ml/src/train_multimodal.py`](file:///Users/eswar.mandapati/Downloads/poxscanner/ml/src/train_multimodal.py).

### 4.1 Two-Stage Training Protocol

To prevent destructive gradient updates from overwriting pretrained visual representations before the multimodal fusion head stabilizes, training is split into two phases:

- **Stage 1 (Backbone Frozen)**:
  - The ResNet-50 visual backbone parameters are frozen (`requires_grad = False`).
  - Only the clinical MLP parameters and late fusion head parameters are updated.
  - Optimizer: `AdamW` ($\text{lr} = 1 \times 10^{-3}$, $\text{weight\_decay} = 1 \times 10^{-4}$).
  - Purpose: Rapid convergence of tabular symptom weights and alignment with the visual embedding space.

- **Stage 2 (End-to-End Fine-Tuning)**:
  - Deeper convolutional residual blocks (`layer3` and `layer4`) of the ResNet-50 backbone are unfrozen.
  - Low-level feature layers (`conv1`, `bn1`, `layer1`, `layer2`) remain frozen to retain generalized edge/texture filters.
  - Optimizer: `AdamW` with differential learning rates:
    - Backbone parameters: $\text{lr} = 1 \times 10^{-4}$
    - Fusion head parameters: $\text{lr} = 5 \times 10^{-4}$
  - Purpose: Domain-specific adaptation of high-level lesion representations in conjunction with clinical cues.

### 4.2 Loss Function & Class Imbalance

To counter the 7.44:1 class imbalance in clinical records:
- **Weighted Cross-Entropy Loss**:
  $$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N w_{y_i} \log \left( \frac{\exp(\ell_{i, y_i})}{\sum_j \exp(\ell_{i, j})} \right)$$
- Weights are set inversely proportional to class frequencies:
  $$w_0 = 1.0, \quad w_1 = \frac{N_{\text{neg}}}{N_{\text{pos}}} \approx 0.134$$
- Ensures minority class errors are not overshadowed by majority class predictions.

---

## 5. Benchmark Performance Matrix

Evaluated on the holdout test set (`test.csv`, $n=270$ images across 12 patient groups):

| Metric | Model A: Image-Only | Model B: Clinical-Only | Model C: Multimodal Fusion | Multimodal Gain vs Image Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 86.67% | 95.19% | **90.37%** | **+3.70%** |
| **Balanced Accuracy** | 82.22% | 87.50% | **84.52%** | **+2.30%** |
| **Precision (Mpox)** | 62.90% | 100.00% | **75.00%** | **+12.10%** |
| **Recall / Sensitivity** | 75.00% | 75.00% | **75.00%** | 0.00% (Preserved) |
| **Specificity** | 89.45% | 100.00% | **94.04%** | **+4.59%** |
| **Macro F1** | 79.99% | 91.41% | **84.52%** | **+4.53%** |
| **Weighted F1** | 87.09% | 94.91% | **90.37%** | **+3.28%** |
| **ROC-AUC** | 91.97% | 97.13% | **93.60%** | **+1.63%** |

### Empirical Insights
1. **False Positive Reduction**: Multimodal fusion decreased false positive predictions from 23 down to 13, raising precision from 62.9% to 75.0%.
2. **Clinical Discrimination**: The tabular symptom features provide strong diagnostic separation when hallmark mucosal or systemic symptoms are present.
3. **Modality Synergy**: Combining visual patterns with clinical presentation produces higher overall accuracy and discriminative confidence than image data alone.

---

## 6. Explainability Methods

The system provides dual-modality explainability in [`ml/src/explainability.py`](file:///Users/eswar.mandapati/Downloads/poxscanner/ml/src/explainability.py), addressing both the visual and clinical dimensions of each inference decision.

### 6.1 Visual Explainability: Grad-CAM

Gradient-Weighted Class Activation Mapping (Grad-CAM) visually highlights the spatial regions of the lesion image that contributed most strongly to the classification.

#### Mathematical Formulation
1. Let $A^k$ represent the $k$-th feature activation map of the target convolutional layer (ResNet-50 `layer4`, $7 \times 7$ spatial grid).
2. Compute the gradient of the predicted class score $y^c$ with respect to activation map $A^k$:
   $$\frac{\partial y^c}{\partial A_{i,j}^k}$$
3. Calculate neuron importance weights $\alpha_k^c$ via global average pooling of gradients:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
4. Compute the class-discriminative localization map as a rectified linear combination:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
5. The map is normalized to $[0.0, 1.0]$, upsampled to $224 \times 224$ via bilinear interpolation, mapped through an OpenCV colormap (`COLORMAP_JET`), and blended over the original photograph ($\alpha = 0.5$).

---

### 6.2 Clinical Explainability: Symptom Attribution (Perturbation Analysis)

To explain the contribution of tabular symptoms to the prediction, the system implements marginal perturbation sensitivity analysis.

#### Mathematical Formulation
1. Let $x_{\text{clin}} = [s_1, s_2, \dots, s_D]$ represent the normalized clinical symptom vector, and let $P(c \mid x_{\text{img}}, x_{\text{clin}})$ represent the predicted probability for class $c$ (where $c = 1$ for Monkeypox).
2. For each clinical symptom $j \in \{1, \dots, D\}$:
   - Compute a baseline counterfactual $x_{\text{clin}}^{(j\to 0)}$ where feature $j$ is ablated to 0 (absence of symptom).
   - Evaluate the perturbed model forward pass:
     $$\Delta P_j = P(c \mid x_{\text{img}}, x_{\text{clin}}) - P(c \mid x_{\text{img}}, x_{\text{clin}}^{(j\to 0)})$$
3. **Direction and Impact**:
   - **Supporting Symptoms ($\Delta P_j > 0$)**: The presence of symptom $j$ increased the model's confidence for the predicted class.
   - **Opposing Symptoms ($\Delta P_j < 0$)**: The presence or absence of symptom $j$ pushed the model away from the predicted class toward the alternative diagnosis.
4. **Ranking**:
   - Results are sorted by $|\Delta P_j|$ into **Top Supporting Symptoms** and **Top Opposing Symptoms** and displayed as discrete quantitative impact cards in the user interface.

---

## 7. Summary of Key Files

| Module / File | Responsibility |
| :--- | :--- |
| [`ml/src/clinical.py`](file:///Users/eswar.mandapati/Downloads/poxscanner/ml/src/clinical.py) | `ClinicalPreprocessor`: Leakage pruning, collinearity filtering, ANOVA feature ranking, imputation, and standardization. |
| [`ml/src/dataset.py`](file:///Users/eswar.mandapati/Downloads/poxscanner/ml/src/dataset.py) | `MultimodalLesionDataset`: Patient-group isolated multi-modal data loader. |
| [`ml/src/architectures/multimodal.py`](file:///Users/eswar.mandapati/Downloads/poxscanner/ml/src/architectures/multimodal.py) | `ClinicalMLP`, `ClinicalOnlyClassifier` (Model B), and `MultimodalClassifier` (Model C). |
| [`ml/src/train_multimodal.py`](file:///Users/eswar.mandapati/Downloads/poxscanner/ml/src/train_multimodal.py) | Staged training pipeline (Stage 1 freeze, Stage 2 unfreeze, class-weighted loss). |
| [`ml/src/compare_models.py`](file:///Users/eswar.mandapati/Downloads/poxscanner/ml/src/compare_models.py) | Automated three-model evaluation generating [`model_comparison.json`](file:///Users/eswar.mandapati/Downloads/poxscanner/ml/artifacts/metrics/model_comparison.json). |
| [`ml/src/explainability.py`](file:///Users/eswar.mandapati/Downloads/poxscanner/ml/src/explainability.py) | Dual explainability: Grad-CAM visual heatmaps and clinical symptom attributions. |
| [`server/app/services/inference.py`](file:///Users/eswar.mandapati/Downloads/poxscanner/server/app/services/inference.py) | Backend inference engine orchestrating multimodal forward passes and explanation payloads. |
| [`client/components/ResultsPanel.tsx`](file:///Users/eswar.mandapati/Downloads/poxscanner/client/components/ResultsPanel.tsx) | Frontend results component displaying predictions, Grad-CAM overlays, and Symptom Attribution cards. |
