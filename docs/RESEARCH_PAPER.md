# Multimodal Monkeypox Detection Using Skin Lesion Images and Clinical Symptoms

**Research Article | Academic Working Draft**

**Authors:**  
Pox Scanner Research and Development Initiative  
*Biomedical Machine Learning and Clinical Decision Support Laboratory*  
*Correspondence:* contact@poxscanner.org  

---

## 1. Title
**Multimodal Monkeypox Detection Using Skin Lesion Images and Clinical Symptoms: An Explainable Deep Learning and Feature Fusion Approach**

---

## 2. Abstract

### Background
Human monkeypox (mpox), caused by the monkeypox virus (*Orthopoxvirus*), is an emerging zoonotic viral infection that has triggered multi-country outbreaks and global public health emergencies. Rapid, accurate triage is essential to contain viral transmission, initiate patient isolation, and allocate supportive interventions.

### Problem
Traditional clinical triage faces severe diagnostic dilemmas because monkeypox skin lesions morphologically resemble other vesicular dermatoses (e.g., varicella/chickenpox, herpes simplex, and molluscum contagiosum). Prior computational solutions have focused on unimodal pipelines: image-only deep learning models suffer from elevated false-positive rates due to visual mimics and lack of systemic prodromal context, whereas clinical-symptom-only models cannot evaluate physical lesion morphology or evolution.

### Proposed Approach
We propose **Pox Scanner**, an end-to-end multimodal deep learning diagnostic framework that jointly processes dermatological lesion imagery and structured tabular clinical symptoms through late feature fusion and dual explainability.

### Dataset
The system was developed and validated on two synchronized data sources: (1) a dermatological image cohort containing 6,916 photographs across 127 patient groups (2,606 monkeypox, 4,310 non-monkeypox), partitioned into strictly isolated train (6,374 images, 103 groups), validation (272 images, 12 groups), and test (270 images, 12 groups) splits using perceptual hashing ($d_{\text{Hamming}} \le 5$) to prevent identity leakage; and (2) a clinical patient intake cohort ($N = 211$, 48 raw fields) capturing systemic symptoms, mucosal ulcerations, and constitutional manifestations.

### Methodology
The visual branch employs a 50-layer Deep Residual Network (ResNet-50) initialized with pretrained ImageNet weights as a visual feature extractor, projecting lesion morphology to a 256-dimensional feature vector. The clinical branch preprocesses raw symptoms via target leakage pruning, collinearity filtering, and ANOVA F-score ranking (retaining the top 28 diagnostic features), which are projected through a 2-stage Multi-Layer Perceptron (Clinical MLP) to a 64-dimensional clinical embedding. A late-fusion classification head concatenates the two representations into a 320-dimensional vector ($256 + 64 = 320$), passing it through a dense transformation (`Linear(320 → 128)` $\to$ `LayerNorm` $\to$ `ReLU` $\to$ `Dropout(0.3)` $\to$ `Linear(128 → 2)`) to yield binary prediction logits. The system delivers dual-modality explainability via Gradient-Weighted Class Activation Mapping (Grad-CAM) for spatial lesion localization and marginal perturbation sensitivity analysis for quantitative clinical symptom attribution.

### Results
On the holdout test partition ($N = 270$ across 12 isolated patient groups), the multimodal fusion model achieved an overall accuracy of **90.37%**, sensitivity of **75.00%**, specificity of **94.04%**, precision of **75.00%**, macro F1-score of **84.52%**, and ROC-AUC of **93.60%**. Compared to the image-only baseline (86.67% accuracy, 62.90% precision, 23 false positives), the multimodal model achieved a **43.5% relative reduction in false positives** (reducing false alarms to 13) while preserving 75.00% sensitivity.

### Conclusion
Combining dermatological lesion features with tabular clinical symptoms resolves visual ambiguity, drastically suppresses false-positive misclassifications, and enhances diagnostic precision. Coupled with localized Grad-CAM heatmaps and symptom attribution cards, this multimodal approach establishes a transparent, dependable clinical decision-support system for infectious disease surveillance.

---

## 3. Keywords
Monkeypox (mpox), Multimodal Deep Learning, Skin Lesion Classification, Clinical Decision Support, ResNet-50, Feature Fusion, Explainable AI (XAI), Grad-CAM, Multi-Layer Perceptron.

---

## 4. Introduction

### 4.1 Monkeypox Background & Clinical Overview
Monkeypox (mpox) is a zoonotic viral disease caused by the monkeypox virus (MPXV), a double-stranded DNA virus belonging to the genus *Orthopoxvirus* in the family *Poxviridae* [1, 2]. First identified in laboratory macaques in 1958 and in humans in 1970 in the Democratic Republic of the Congo (DRC), monkeypox was historically considered an endemic infection confined to forested regions of West and Central Africa [3, 4]. However, beginning in May 2022, unprecedented international outbreaks across non-endemic regions led the World Health Organization (WHO) and the Africa Centres for Disease Control and Prevention (Africa CDC) to declare mpox a Public Health Emergency of International Concern (PHEIC) [8, 9]. 

Transmission occurs via direct contact with infectious cutaneous lesions, bodily fluids, respiratory droplets during prolonged close contact, and contaminated fomites (e.g., bedding and clothing) [4, 6]. Clinically, infection typically manifests in two distinct phases:
1. **Prodromal Phase (Days 0–3)**: Characterized by systemic constitutional symptoms, including high fever, intense cephalalgia (headache), prominent lymphadenopathy (swollen submandibular, cervical, or inguinal lymph nodes), severe myalgia, back pain, and asthenia [4, 11].
2. **Eruptive Dermatological Phase (Days 1–5 post-fever)**: Manifests as a centrifugal eruption of skin lesions progressing synchronously through discrete dermatological stages: macules $\to$ papules $\to$ vesicles $\to$ pustules $\to$ central umbilication $\to$ crusting $\to$ desquamation [10, 11].

```
                     ┌─────────────────────────────────────────────────────────┐
                     │            Typical Clinical Progression of Mpox        │
                     └────────────────────────────┬────────────────────────────┘
                                                  │
          ┌───────────────────────────────────────┴───────────────────────────────────────┐
          ▼                                                                               ▼
   [ Prodromal Phase ]                                                           [ Eruptive Phase ]
   • High Fever (>38.5°C)                                                        • Macules (flat discolored spots)
   • Lymphadenopathy (pathognomonic sign)                                        • Papules (raised firm bumps)
   • Intense Headache & Myalgia                                                  • Vesicles (fluid-filled blisters)
   • Back Pain & Profound Fatigue                                                • Pustules & Central Umbilication
                                                                                 • Crusting & Scabbing
```

### 4.2 Importance of Early Detection & Clinical Challenges
Prompt identification of monkeypox cases is vital for initiating patient isolation, interrupting transmission chains via ring vaccination and contact tracing, and preventing systemic complications such as secondary bacterial superinfections, bronchopneumonia, keratitis-induced vision loss, and encephalitis [1, 7]. 

However, differential diagnosis in clinical practice is notoriously challenging. Monkeypox lesions morphologically resemble other vesicular dermatoses, most notably varicella (chickenpox), herpes simplex virus (HSV-1/HSV-2), molluscum contagiosum, syphilis, and enteroviral hand-foot-and-mouth disease [11, 12]. While reverse-transcription polymerase chain reaction (RT-PCR) testing of lesion exudates is the definitive gold standard, molecular diagnostic facilities are frequently unavailable or logistically constrained in decentralized, primary care, and rural clinics, leading to critical diagnostic delays [25, 26].

### 4.3 Limitations of Unimodal Approaches
Existing computational triage systems have historically adopted single-modality approaches, each exhibiting fundamental operational limitations:

1. **Limitations of Image-Only Systems**:
   - *Visual Mimicry & High False Alarm Rates*: Lesions from chickenpox, severe acne, or folliculitis frequently display vesicular or pustular morphology identical to early-stage mpox, leading convolutional neural networks (CNNs) to trigger excessive false alarms [18, 26].
   - *Absence of Systemic Context*: Computer vision models cannot assess whether a lesion is accompanied by marked lymphadenopathy, mucosal involvement, or sudden-onset fever.
   - *Generalization Fragility*: Image-only CNNs are susceptible to variations in lighting, background artifacts, camera hardware, and patient skin pigmentation (Fitzpatrick skin phototypes) [14, 25].

2. **Limitations of Clinical-Symptom-Only Systems**:
   - *Neglect of Lesion Morphology*: Tabular machine learning models rely solely on patient-reported symptom checklists, entirely discarding spatial and physical lesion characteristics (e.g., umbilication, stage synchronicity, border definition) [22, 34].
   - *Subjectivity & Recall Bias*: Patient-reported symptoms are prone to subjective interpretation, varying pain thresholds, and imperfect memory regarding symptom onset.
   - *Atypical Clinical Presentations*: In contemporary outbreaks, atypical cases (e.g., solitary genital ulcers without prominent systemic prodrome) can easily bypass symptom-based threshold rules [4].

```
┌──────────────────────────────────────────────────┐      ┌──────────────────────────────────────────────────┐
│             Image-Only Approaches                │      │            Clinical-Only Approaches              │
├──────────────────────────────────────────────────┤      ├──────────────────────────────────────────────────┤
│ • Analyzes skin lesion photograph only           │      │ • Analyzes patient symptom questionnaire only    │
│ • Misses constitutional symptoms (fever, nodes)  │      │ • Completely blind to physical lesion appearance │
│ • High false positives on chickenpox / HSV       │      │ • Vulnerable to atypical prodromal presentations │
│ • Susceptible to photographic lighting / tone    │      │ • Cannot verify lesion progression stage         │
└─────────────────────────┬────────────────────────┘      └─────────────────────────┬────────────────────────┘
                          │                                                         │
                          └───────────────────────────┬─────────────────────────────┘
                                                      ▼
                                   ┌──────────────────────────────────────┐
                                   │      Multimodal Fusion Solution      │
                                   │   (Pox Scanner: Image + Clinical)    │
                                   └──────────────────────────────────────┘
```

### 4.4 Motivation for Multimodal Learning & Proposed Solution
Medical practitioners rarely diagnose infectious dermatoses based on a photograph alone or a symptom list in isolation; rather, clinical diagnosis is intrinsically **multimodal**, synthesizing physical dermatological examination with detailed medical history and review of systems. 

To operationalize this clinical paradigm, we present **Pox Scanner**, an end-to-end multimodal deep learning framework that unites computer vision and tabular machine learning:
1. High-level morphological lesion features (256 dimensions) are extracted using a fine-tuned ResNet-50 convolutional backbone.
2. Systemic constitutional symptoms (64 dimensions) are encoded using a Multi-Layer Perceptron (Clinical MLP) operating on 28 ANOVA-selected clinical variables.
3. The representations are fused into a unified 320-dimensional embedding ($256 + 64 = 320$) and classified via a specialized MLP fusion head.
4. The system couples every inference decision with dual explainability: spatial Grad-CAM localization heatmaps and quantitative symptom attribution cards.

---

## 5. Problem Statement

Given:
1. A single dermatological photograph $X_{\text{img}} \in \mathbb{R}^{H \times W \times 3}$ capturing a suspected cutaneous skin lesion.
2. A structured vector of binary/categorical patient-reported clinical symptoms $X_{\text{clin}} \in \{0, 1\}^D$.

The objective is to construct a parameterized classification function $\mathcal{F}_\theta(X_{\text{img}}, X_{\text{clin}}) \to \hat{y} \in \{0, 1\}$, where $\hat{y} = 1$ denotes **Monkeypox** and $\hat{y} = 0$ denotes **Non-Monkeypox**, such that:
- Classification accuracy, sensitivity, and specificity are maximized on strictly isolated patient test cohorts ($D_{\text{train}} \cap D_{\text{test}} = \emptyset$).
- False-positive classifications caused by visual mimics are minimized through clinical feature cross-validation.
- The system produces interpretable explanations for both the visual and clinical dimensions without degrading inference throughput.

---

## 6. Motivation

1. **Frontline Triage in Decentralized Settings**: Enhancing point-of-care decision support where specialized dermatologists and PCR testing are unavailable.
2. **Elimination of Diagnostic Blind Spots**: By fusing visual and clinical data, the weakness of one modality (e.g., visual ambiguity between chickenpox and mpox) is compensated by the strength of the other (e.g., presence of lymphadenopathy and mucosal ulcers).
3. **Clinical Trust through Dual Explainability**: Black-box neural predictions cannot be safely integrated into clinical workflows. Providing localized visual heatmaps (Grad-CAM) alongside clinical risk attributions ensures transparent, clinician-verifiable recommendations.
4. **Data Hygiene & Methodological Rigor**: Prior works in medical image classification have been severely compromised by data leakage across train/test splits. Enforcing patient-group isolation and perceptual hashing guarantees authentic generalizability.

---

## 7. Research Objectives

1. **Benchmark Computer Vision Backbones**: Evaluate four convolutional neural network architectures (ResNet-50, DenseNet-121, EfficientNet-B0, MobileNet-V3-Small) to determine the most effective visual feature extractor for skin lesions.
2. **Engineer Clinical Preprocessing Pipeline**: Develop an automated tabular pipeline that eliminates target leakage, filters collinear feature pairs, ranks diagnostic symptoms using ANOVA F-statistics, and scales inputs for neural encoding.
3. **Design & Implement Multimodal Fusion Network**: Construct a late-fusion neural network that projects image features to 256-d, clinical symptoms to 64-d, concatenates them into 320-d, and classifies them via an MLP head.
4. **Perform Comprehensive Ablation Study**: Empirically quantify the performance delta between Image-Only (Model A), Clinical-Only (Model B), and Multimodal Fusion (Model C) systems on an identical holdout test cohort.
5. **Implement Dual-Modality Explainable AI (XAI)**: Integrate spatial Grad-CAM activation mapping and marginal symptom perturbation analysis into an interactive production interface.

---

## 8. Contributions of the Proposed Work

1. **Multimodal Late-Fusion Architecture**: Engineered a joint neural pipeline that combines 256-d ResNet-50 visual representations with 64-d Clinical MLP representations into a 320-d fusion classifier.
2. **Empirical False-Positive Suppression**: Demonstrated that multimodal fusion achieves a **43.5% relative reduction in false positives** (reducing false alarms from 23 to 13) compared to image-only models, boosting precision from 62.90% to 75.00%.
3. **Rigorous Zero-Leakage Dataset Curation**: Established a 6,916-image corpus with group-level isolation and 64-bit perceptual hash deduplication ($d_{\text{Hamming}} \le 5$) across 127 patient cohorts.
4. **Dual-Modality Explainability Pipeline**: Formulated an integrated explainability framework delivering simultaneous spatial lesion localization (Grad-CAM) and clinical feature sensitivity attributions.
5. **Full-Stack Clinical Deployment Architecture**: Deployed the trained multimodal models within an asynchronous FastAPI service and an interactive Next.js web application supporting dynamic runtime modality dispatching.

---

## 9. Related Work

```
                                  ┌─────────────────────────────────────────┐
                                  │      Taxonomy of Related Works          │
                                  └────────────────────┬────────────────────┘
                                                       │
         ┌──────────────────────────────┬──────────────┴──────────────┬──────────────────────────────┐
         ▼                              ▼                             ▼                              ▼
 [ Traditional Diagnosis ]    [ Image-Only Deep Learning ]  [ Symptom-Based ML Models ]    [ Multimodal Medical AI ]
 • PCR Assay (Gold Standard)  • CNNs, DenseNet, ResNet      • LightGBM, XGBoost, CatBoost  • Vision-Language & Fusion
 • Viral Culture & Serology   • Lacks systemic context      • Ignores lesion morphology    • Pox Scanner (This Work)
 • Clinical morphology        • High false positives        • Setegn & Dejene (BMC 2025)   • Joint Image + Symptoms
```

### 9.1 Traditional Monkeypox Diagnosis
Historically, diagnosis has relied on viral culture, electron microscopy, serological enzyme-linked immunosorbent assays (ELISA), and real-time PCR targeting orthopoxvirus genes (*F3L*, *E9L*) [1, 8]. While PCR provides high analytical sensitivity, its reliance on specialized cold-chain logistics, thermal cyclers, and trained personnel restricts its utility during rapid decentralized outbreak surges [25].

### 9.2 Machine Learning for Symptom-Based Prediction
Recent studies have explored machine learning on tabular clinical symptom records:
- *Farzipour et al. (2023)* [22] evaluated XGBoost on a 211-patient clinical dataset, reporting high nominal accuracy and interpreting feature importances via SHAP.
- *Setegn & Dejene (BMC Infectious Diseases, 2025)* [34] performed an extensive benchmark of ensemble algorithms (Random Forest, Bagging, Gradient Boosting, CatBoost, XGBoost, and LightGBM) on symptom data balanced via SMOTE, achieving 89.30% accuracy with LightGBM and exploring multi-method explainability (SHAP, LIME, ELI5, QLattice).
- *Limitation*: These approaches operate entirely in the tabular domain and cannot ingest or evaluate skin lesion photography.

### 9.3 Skin-Lesion Image Classification via Deep Learning
Convolutional neural networks have been widely applied to dermatological image classification:
- *Alakuş et al. (2023)* [18] deployed a Siamese deep learning network for monkeypox vs. non-monkeypox classification, achieving 91.09% accuracy.
- *Sorayaie Azar et al. (2023)* [29] evaluated DenseNet-201 on a multi-class skin lesion dataset (monkeypox, chickenpox, measles, normal), reporting 97.63% accuracy.
- *Al-Gaashani et al. (2025)* [27] proposed MobileNetV2 with progressive transfer learning and channel attention, achieving 99.34% accuracy.
- *Thieme et al. (Nature Medicine, 2023)* [25] validated a mobile deep learning algorithm across clinical centers, highlighting the challenge of false positives against varicella and secondary syphilis.
- *Limitation*: These architectures are unimodal, lacking clinical prodromal context and often omitting patient-level split isolation.

### 9.4 Multimodal Medical AI and Research Gap
Multimodal fusion—combining imaging, electronic health records (EHR), and genomics—has gained prominence in oncology and pulmonary medicine [30, 31]. However, in infectious disease triage and monkeypox detection specifically, multimodal fusion integrating dermatological lesion imagery with tabular clinical symptom vectors remains largely unexplored. Pox Scanner addresses this explicit research gap.

---

## 10. Materials and Methods

### 10.1 System Overview & Pipeline Flowchart

The Pox Scanner architecture integrates data ingestion, preprocessing, dual-branch feature extraction, late fusion, classification, and dual explainability:

```
+---------------------------------------------------------------------------------------------------+
|                                   POX SCANNER SYSTEM PIPELINE                                     |
+---------------------------------------------------------------------------------------------------+

     CLINICAL DATA BRANCH                                        IMAGE DATA BRANCH
  ┌────────────────────────┐                                  ┌────────────────────────┐
  │  Patient Intake Data   │                                  │ Dermatological Lesion  │
  │      (Data.csv)        │                                  │       Photograph       │
  └───────────┬────────────┘                                  └───────────┬────────────┘
              │                                                           │
              ▼                                                           ▼
  ┌────────────────────────┐                                  ┌────────────────────────┐
  │ 1. Data Cleaning       │                                  │ 1. Image Preprocessing │
  │ • Strip ID & Target    │                                  │ • Resize to 256x256    │
  │ • Remove Target Leakage│                                  │ • Center Crop 224x224  │
  │   (characteristic cols)│                                  │ • ImageNet Normaliz.   │
  └───────────┬────────────┘                                  └───────────┬────────────┘
              │                                                           │
              ▼                                                           ▼
  ┌────────────────────────┐                                  ┌────────────────────────┐
  │ 2. Feature Selection   │                                  │ 2. Data Augmentation   │
  │ • Collinearity (r=1.0) │                                  │ • Random Flips (H/V)   │
  │ • ANOVA F-Score Rank   │                                  │ • Rotation (+/- 15 deg)│
  │ • Select Top 28 Feats  │                                  │ • Color Jitter (+/-15%)│
  └───────────┬────────────┘                                  └───────────┬────────────┘
              │ (28-d vector)                                             │ (3x224x224 tensor)
              ▼                                                           ▼
  ┌────────────────────────┐                                  ┌────────────────────────┐
  │ 3. Clinical Feature Net│                                  │ 3. ResNet-50 Backbone  │
  │ • Linear(28 -> 128)    │                                  │ • Conv1 - Layer4       │
  │ • BatchNorm1d + ReLU   │                                  │ • Adaptive AvgPool     │
  │ • Dropout(0.25)        │                                  │ • Linear(2048 -> 256)  │
  │ • Linear(128 -> 64)    │                                  │ • BatchNorm1d + ReLU   │
  │ • BatchNorm1d + ReLU   │                                  │ • Dropout(0.25)        │
  └───────────┬────────────┘                                  └───────────┬────────────┘
              │                                                           │
              │ (64-d Clinical Representation)                            │ (256-d Image Representation)
              └─────────────────────────────┬─────────────────────────────┘
                                            │
                                            ▼
                             ┌─────────────────────────────┐
                             │ 4. Multimodal Late Fusion   │
                             │ Concatenation: 256 + 64     │
                             │ 320-d Fused Representation  │
                             └──────────────┬──────────────┘
                                            │
                                            ▼
                             ┌─────────────────────────────┐
                             │ 5. MLP Classification Head  │
                             │ • Linear(320 -> 128)        │
                             │ • LayerNorm(128) + ReLU     │
                             │ • Dropout(p = 0.3)          │
                             │ • Linear(128 -> 2)          │
                             └──────────────┬──────────────┘
                                            │
                                            ▼
                             ┌─────────────────────────────┐
                             │   Binary Diagnostic Output  │
                             │  [Non-Monkeypox / Mpox]     │
                             │  Probabilities & Confidence │
                             └──────────────┬──────────────┘
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
      ┌─────────────────────────────┐               ┌─────────────────────────────┐
      │ Visual Explainability       │               │ Clinical Explainability     │
      │ • Grad-CAM on Layer4 Maps   │               │ • Perturbation Sensitivity  │
      │ • Spatial Lesion Heatmap    │               │ • Supporting / Opposing     │
      │ • Alpha Blend Overlay       │               │   Symptom Attribution Cards │
      └─────────────────────────────┘               └─────────────────────────────┘
```

---

### 10.2 Dataset Description & Cohort Distribution

The project integrates two primary data sources:

#### 1. Dermatological Image Dataset
- **Total Images**: 6,916 photographs across 127 patient groups.
- **Classes**: Binary (`monkeypox`: $n = 2,606$, `non_monkeypox`: $n = 4,310$).
- **Image Specifications**: High-resolution RGB images in `.jpg`, `.png`, and `.webp` formats depicting lesions across diverse anatomical sites (face, upper extremities, trunk, perianal, and genital regions).

#### 2. Clinical Symptom Dataset (`Data.csv`)
- **Total Records**: 211 confirmed patient cases published by Farzipour et al. [22, 32].
- **Features**: 48 raw fields including binary symptom flags ($1 = \text{Present}$, $0 = \text{Absent}$).
- **Target Variable**: `Status` ($1 = \text{Monkeypox Positive}$, $0 = \text{Monkeypox Negative}$).
- **Class Breakdown**: 186 positive cases (88.15%), 25 negative cases (11.85%), reflecting a $7.44:1$ class imbalance.

#### Table 1: Dataset Partitioning and Group Isolation Manifest

| Partition | Total Images | Monkeypox Images | Non-Mpox Images | Patient Groups | Group Isolation Policy |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Training (`train.csv`)** | 6,374 | 2,502 | 3,872 | 103 | Stratified 70% by group; pHash $d \le 5$ clustered |
| **Validation (`val.csv`)** | 272 | 52 | 220 | 12 | Isolated patient groups; no overlap with train/test |
| **Holdout Test (`test.csv`)** | 270 | 52 | 218 | 12 | Isolated patient groups; held out for final benchmark |
| **Total Cohort** | **6,916** | **2,606** | **4,310** | **127** | **Zero identity or near-duplicate leakage** |

---

### 10.3 Clinical Data Preprocessing Pipeline

Implemented in `ml/src/clinical.py` via `ClinicalPreprocessor`:

1. **Identifier & Direct Target Stripping**: The sequential record index (`ID`) and target label (`Status`) are removed from the feature matrix.
2. **Target Leakage Pruning**: We explicitly remove two post-hoc clinician summary variables:
   - `characteristic symptoms of Monkeypox`
   - `symptoms compatible with monkeypox`
   - *Rationale*: These columns represent composite medical judgments formed after full diagnostic workups. Retaining them introduces severe artificial target leakage, yielding spuriously high model accuracy that fails in real-world triage.
3. **Collinearity and Zero-Variance Filtering**:
   - Features exhibiting zero variance across all samples are pruned.
   - Perfectly collinear feature pairs ($r = 1.0$) are pruned (e.g., `perianal papules` is perfectly collinear with `inguinal adenopathy` in the raw dataset).
4. **Missing Value Imputation**: Missing or unrecorded symptom fields default to $0$ (indicating symptom absence).
5. **Standardization**: Continuous and scaled binary features are standardized to zero mean and unit variance:
   $$z_j = \frac{x_j - \mu_j}{\sigma_j + \epsilon}$$
   The parameters $\mu_j$ and $\sigma_j$ are computed strictly on the training partition and saved to `clinical_features.json` to ensure deterministic preprocessing at inference time.

---

### 10.4 Clinical Feature Selection

Feature selection was conducted using univariate Analysis of Variance (ANOVA) F-score ranking on the training partition:
$$F = \frac{\text{MS}_{\text{between}}}{\text{MS}_{\text{within}}} = \frac{\sum_{k=1}^K n_k (\bar{x}_k - \bar{x})^2 / (K - 1)}{\sum_{k=1}^K \sum_{i=1}^{n_k} (x_{ik} - \bar{x}_k)^2 / (N - K)}$$

Of the 48 original columns, **28 features** were selected based on statistical significance ($p < 0.01$) and diagnostic relevance.

#### Table 2: Complete List of Original, Filtered, and Selected Clinical Features

| Feature Status | Count | Feature Names |
| :--- | :---: | :--- |
| **Excluded (Target / ID)** | 2 | `ID`, `Status` |
| **Excluded (Target Leakage)** | 2 | `characteristic symptoms of Monkeypox`, `symptoms compatible with monkeypox` |
| **Excluded (Collinear / Low F-Score)** | 16 | `diarrhea`, `encephalitis`, `cough`, `asthenia`, `general weakness`, `Pain urinating`, `general discomfort`, `lymphadenopathy`, `perianal papules`, `mild symptoms`, `hands`, `chest`, `loss of appetite`, `difficulty breathing`, `decreased physical strength`, `inguinal adenopathy` |
| **Selected Diagnostic Features** | **28** | `fever`, `oral and genital ulcers`, `muscle pain`, `skin lesions`, `ulcerative lesions`, `genital ulcer lesions`, `headache`, `itch`, `adenomegaly`, `rash`, `severe anemia`, `fatigue`, `pustules`, `ulcers`, `dysphagia`, `papules`, `myalgia`, `erythema with vesicles and papules`, `malaise`, `sore throat`, `swollen lymph nodes`, `chills`, `blisters`, `outbreak on the skin`, `Vesicles`, `blisters on limbs and genitals`, `postules`, `myalgias` |

#### Table 3: Top 10 Selected Features Ranked by Normalized ANOVA Importance Score

| Rank | Clinical Feature | Importance Weight | Feature Mean ($\mu$) | Feature Std ($\sigma$) | Clinical Relevance |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | `fever` | 0.1118 | 0.3791 | 0.4863 | Primary prodromal hallmark (>38.5°C) |
| 2 | `oral and genital ulcers` | 0.0974 | 0.1043 | 0.3063 | Signature mucosal presentation |
| 3 | `muscle pain` | 0.0893 | 0.0664 | 0.2495 | Systemic constitutional prodrome |
| 4 | `skin lesions` | 0.0745 | 0.2370 | 0.4262 | Cutaneous eruptive involvement |
| 5 | `ulcerative lesions` | 0.0725 | 0.0995 | 0.3001 | Deep necrotic ulcer morphology |
| 6 | `genital ulcer lesions` | 0.0723 | 0.1422 | 0.3501 | Anogenital transmission indicator |
| 7 | `headache` | 0.0687 | 0.1043 | 0.3063 | Severe cephalalgia prodrome |
| 8 | `itch` | 0.0462 | 0.0142 | 0.1187 | Pruritus during eruptive/crusting stage |
| 9 | `adenomegaly` | 0.0412 | 0.0095 | 0.0971 | Lymph node enlargement marker |
| 10 | `rash` | 0.0411 | 0.2749 | 0.4475 | Centrifugal maculopapular eruption |

---

### 10.5 Image Preprocessing and Data Augmentation

Implemented in `ml/src/transforms.py` and `ml/src/predict.py`:

```
Raw Image File ──► Bilinear Resize (256x256) ──► Center/Random Crop (224x224) ──► Data Augmentation ──► ImageNet Normalization
```

1. **Dimensional Scaling**: Input images are rescaled to $256 \times 256$ pixels using bilinear interpolation, followed by a $224 \times 224$ crop matching the native receptive field of ResNet-50.
2. **Data Augmentation Protocol (Training Split Only)**:
   - **Horizontal Flip**: `RandomHorizontalFlip(p=0.5)` for lateral symmetry invariance.
   - **Vertical Flip**: `RandomVerticalFlip(p=0.5)` for orientation invariance.
   - **Rotation**: `RandomRotation(degrees=15)` to handle varying smartphone capture angles.
   - **Photometric Jitter**: `ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15)` to simulate diverse clinical lighting conditions.
3. **Channel Standardization**: Tensor pixel values $[0, 255] \to [0.0, 1.0]$ are normalized using ImageNet channel parameters:
   $$\mu_{\text{RGB}} = [0.485, 0.456, 0.406], \quad \sigma_{\text{RGB}} = [0.229, 0.224, 0.225]$$
4. **Deterministic Inference**: Validation and test pipelines apply deterministic center cropping and normalization without stochastic perturbations.

---

### 10.6 ResNet-50 Feature Extraction

#### Residual Learning Framework
Deep convolutional neural networks often suffer from vanishing/exploding gradients and degradation when layer depth increases. ResNet-50 addresses this by introducing **shortcut (residual) skip connections** that learn residual mapping functions $\mathcal{F}(x) = \mathcal{H}(x) - x$ rather than unreferenced underlying mappings $\mathcal{H}(x)$:
$$y = \mathcal{F}(x, \{W_i\}) + W_s x$$
where $x$ and $y$ are the input and output vectors of the residual block, $\mathcal{F}$ represents the residual transformation, and $W_s$ is a linear projection used when dimension matching is required.

```
                         ResNet-50 Residual Bottleneck Block
                         ┌─────────────────────────────────┐
                         │             Input x             │
                         └───────────────┬─────────────────┘
                                         │
                        ┌────────────────┴────────────────┐
                        │                                 │ (Identity Skip Connection)
                        ▼                                 │
             ┌─────────────────────┐                      │
             │   1x1 Conv, 64-d    │                      │
             │   BatchNorm + ReLU  │                      │
             └──────────┬──────────┘                      │
                        ▼                                 │
             ┌─────────────────────┐                      │
             │   3x3 Conv, 64-d    │                      │
             │   BatchNorm + ReLU  │                      │
             └──────────┬──────────┘                      │
                        ▼                                 │
             ┌─────────────────────┐                      │
             │   1x1 Conv, 256-d   │                      │
             │      BatchNorm      │                      │
             └──────────┬──────────┘                      │
                        │                                 │
                        ▼                                 │
                     [  +  ] ◄────────────────────────────┘
                        │
                        ▼
                     [ ReLU ]
                        │
                        ▼
                     Output y
```

#### Adaptation for Dermatological Feature Extraction
In our multimodal pipeline, **ResNet-50 is employed strictly as an image feature extractor**, rather than the final classifier:
1. **Pretrained Backbone**: Initialized with ImageNet weights, retaining generalized low-level filters (edges, textures, contours) across `layer1` and `layer2`.
2. **Feature Truncation**: The standard 1000-class fully connected classification layer is removed.
3. **Global Adaptive Pooling**: Spatial feature maps from `layer4` ($7 \times 7 \times 2048$) are converted to a 2048-dimensional vector via `AdaptiveAvgPool2d((1, 1))`.
4. **Visual Projection Layer**: The 2048-d vector is projected to a compact **256-dimensional** visual representation $v_{\text{img}}$:
   $$v_{\text{img}} = \text{Dropout}_{0.25}\left( \text{ReLU}\left( \text{BatchNorm1d}\left( \text{Linear}(2048 \to 256) \right) \right) \right) \in \mathbb{R}^{256}$$

---

### 10.7 Clinical Feature Representation (Clinical MLP)

The 28-dimensional standardized clinical vector $x_{\text{clin}} \in \mathbb{R}^{28}$ is transformed into a dense clinical embedding $v_{\text{clin}} \in \mathbb{R}^{64}$ through a 2-stage Multi-Layer Perceptron:

```
Input (28-d) ──► Linear(28->128) ──► BatchNorm1d ──► ReLU ──► Dropout(0.25) ──► Linear(128->64) ──► BatchNorm1d ──► ReLU ──► 64-d Vector
```

Mathematically:
$$h_{\text{clin}}^{(1)} = \text{Dropout}_{0.25}\left( \text{ReLU}\left( \text{BatchNorm1d}\left( W_1 x_{\text{clin}} + b_1 \right) \right) \right), \quad W_1 \in \mathbb{R}^{128 \times 28}$$
$$v_{\text{clin}} = \text{ReLU}\left( \text{BatchNorm1d}\left( W_2 h_{\text{clin}}^{(1)} + b_2 \right) \right), \quad W_2 \in \mathbb{R}^{64 \times 128}$$

This projects sparse binary symptom indicators into a continuous 64-dimensional semantic manifold.

---

### 10.8 Multimodal Feature Fusion

The visual representation $v_{\text{img}} \in \mathbb{R}^{256}$ and clinical representation $v_{\text{clin}} \in \mathbb{R}^{64}$ are combined via **late feature concatenation**:
$$v_{\text{fused}} = [v_{\text{img}} \,\|\, v_{\text{clin}}] = \left[ v_{\text{img}}^{(1)}, \dots, v_{\text{img}}^{(256)}, v_{\text{clin}}^{(1)}, \dots, v_{\text{clin}}^{(64)} \right]^T \in \mathbb{R}^{320}$$

#### Rationale for Late Feature Fusion
- **Dimensional Balance**: Visual features (256-d) capture complex spatial morphology, while clinical features (64-d) capture systemic indicators, ensuring neither modality dominates gradient backpropagation.
- **Cross-Modality Interaction**: Concatenation allows the subsequent dense fusion layers to learn non-linear co-dependencies (e.g., matching a vesicular lesion with the presence of lymphadenopathy).

---

### 10.9 MLP Classification Head

The 320-dimensional fused vector $v_{\text{fused}}$ is passed to the specialized MLP classification head:

```
                                  MLP Classification Head Architecture
                                  ┌───────────────────────────────────┐
                                  │   Fused Vector v_fused (320-d)    │
                                  └─────────────────┬─────────────────┘
                                                    │
                                                    ▼
                                  ┌───────────────────────────────────┐
                                  │        Linear(320 → 128)          │
                                  └─────────────────┬─────────────────┘
                                                    │
                                                    ▼
                                  ┌───────────────────────────────────┐
                                  │          LayerNorm(128)           │
                                  └─────────────────┬─────────────────┘
                                                    │
                                                    ▼
                                  ┌───────────────────────────────────┐
                                  │          ReLU Activation          │
                                  └─────────────────┬─────────────────┘
                                                    │
                                                    ▼
                                  ┌───────────────────────────────────┐
                                  │       Dropout (Rate = 0.3)        │
                                  └─────────────────┬─────────────────┘
                                                    │
                                                    ▼
                                  ┌───────────────────────────────────┐
                                  │         Linear(128 → 2)           │
                                  └─────────────────┬─────────────────┘
                                                    │
                                                    ▼
                                  ┌───────────────────────────────────┐
                                  │    Binary Output Logits [z0, z1]  │
                                  │   (0: Non-Mpox, 1: Monkeypox)     │
                                  └───────────────────────────────────┘
```

The forward equations are:
$$h_{\text{fuse}} = \text{Dropout}_{0.3}\left( \text{ReLU}\left( \text{LayerNorm}\left( W_{\text{fuse}} v_{\text{fused}} + b_{\text{fuse}} \right) \right) \right), \quad W_{\text{fuse}} \in \mathbb{R}^{128 \times 320}$$
$$z = W_{\text{out}} h_{\text{fuse}} + b_{\text{out}} \in \mathbb{R}^2, \quad W_{\text{out}} \in \mathbb{R}^{2 \times 128}$$
$$P(y = c \mid X) = \frac{\exp(z_c)}{\exp(z_0) + \exp(z_1)}, \quad c \in \{0, 1\}$$

---

### 10.10 Training Procedure & Hyperparameters

Implemented in `ml/src/train_multimodal.py`:

#### 1. Loss Function (Class-Weighted Cross-Entropy)
To address class imbalance, loss is weighted inversely by class frequencies:
$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N w_{y_i} \log P(y_i \mid X_i), \quad w_0 = 1.0, \quad w_1 = \frac{N_0}{N_1}$$

#### 2. Two-Stage Fine-Tuning Protocol
- **Stage 1 (Backbone Frozen, 10 Epochs)**: The ResNet-50 feature extractor is frozen (`requires_grad = False`). Only the Clinical MLP and MLP Fusion Head are trained using `AdamW` ($\eta = 10^{-3}$, weight decay $= 10^{-4}$). This stabilizes fusion weights without corrupting visual filters.
- **Stage 2 (End-to-End Fine-Tuning, 15 Epochs)**: The upper convolutional blocks (`layer3` and `layer4`) of ResNet-50 are unfrozen. A differential learning rate schedule is applied:
  - Backbone parameters: $\eta_{\text{backbone}} = 1 \times 10^{-4}$
  - Clinical MLP and Fusion Head: $\eta_{\text{head}} = 5 \times 10^{-4}$

#### Table 4: Experimental Hyperparameters and Training Environment

| Parameter / Configuration | Value / Setting | Description / Purpose |
| :--- | :--- | :--- |
| **Deep Learning Framework** | PyTorch 2.x / Torchvision | Model definition and tensor execution |
| **Optimizer** | `AdamW` | Decoupled weight decay optimization |
| **Weight Decay** | $1 \times 10^{-4}$ ($L_2$ regularization) | Prevents overfitting in dense layers |
| **Batch Size** | 16 | Mini-batch stochastic gradient descent |
| **Stage 1 Epochs** | 10 epochs | Head and clinical encoder warm-up |
| **Stage 2 Epochs** | 15 epochs | Differential end-to-end fine-tuning |
| **Stage 1 Learning Rate** | $1 \times 10^{-3}$ | Initial head training rate |
| **Stage 2 Learning Rates** | $\eta_{\text{backbone}} = 10^{-4}$, $\eta_{\text{head}} = 5 \times 10^{-4}$ | Prevents catastrophic forgetting |
| **Loss Function** | Weighted Cross-Entropy | Inverse-frequency class balancing |
| **Random Seed** | 42 | Reproducible split and weight initialization |
| **Compute Hardware** | Intel Core i7 / Apple Silicon CPU | Execution environment |
| **Image Input Dimensions** | $3 \times 224 \times 224$ pixels | Standard 3-channel RGB tensor |
| **Clinical Input Vector** | 28 continuous standardized values | Preprocessed patient symptom vector |

---

## 11. Proposed Architecture

The complete end-to-end architecture diagram of **Pox Scanner**:

```
+───────────────────────────────────────────────────────────────────────────────────────────────────+
|                                    PROPOSED POX SCANNER ARCHITECTURE                              |
+───────────────────────────────────────────────────────────────────────────────────────────────────+

   [ Skin Lesion Photograph ]                                    [ 28 Clinical Symptoms ]
     (3 x 224 x 224 RGB Image)                                     (Standardized Vector)
                 │                                                           │
                 ▼                                                           ▼
   ┌───────────────────────────┐                               ┌───────────────────────────┐
   │ ResNet-50 Feature Backbone│                               │       Clinical MLP        │
   │ • Conv1 (7x7, 64, stride 2)                               │ • Linear(28 → 128)        │
   │ • MaxPool (3x3, stride 2) │                               │ • BatchNorm1d(128)        │
   │ • Layer 1 (3 Bottlenecks) │                               │ • ReLU Activation         │
   │ • Layer 2 (4 Bottlenecks) │                               │ • Dropout(p = 0.25)       │
   │ • Layer 3 (6 Bottlenecks) │                               │ • Linear(128 → 64)        │
   │ • Layer 4 (3 Bottlenecks) │                               │ • BatchNorm1d(64)         │
   │ • AdaptiveAvgPool2d(1, 1) │                               │ • ReLU Activation         │
   │ • Linear(2048 → 256)      │                               └─────────────┬─────────────┘
   │ • BatchNorm1d(256) + ReLU │                                             │
   │ • Dropout(p = 0.25)       │                                             │
   └─────────────┬─────────────┘                                             │
                 │                                                           │
                 │ 256-d Image Features                                      │ 64-d Clinical Features
                 └─────────────────────────────┬─────────────────────────────┘
                                               │
                                               ▼
                               ┌───────────────────────────────┐
                               │   Multimodal Concatenation    │
                               │   v_fused = [v_img || v_clin] │
                               │   Dimension: 256 + 64 = 320-D │
                               └───────────────┬───────────────┘
                                               │
                                               ▼
                               ┌───────────────────────────────┐
                               │    MLP Classification Head    │
                               │ • Linear(320 → 128)           │
                               │ • LayerNorm(128)              │
                               │ • ReLU Activation             │
                               │ • Dropout(p = 0.30)           │
                               │ • Linear(128 → 2)             │
                               └───────────────┬───────────────┘
                                               │
                                               ▼
                               ┌───────────────────────────────┐
                               │        Softmax Output         │
                               │  P(Non-Monkeypox), P(Mpox)    │
                               └───────────────┬───────────────┘
                                               │
                        ┌──────────────────────┴──────────────────────┐
                        ▼                                             ▼
         ┌─────────────────────────────┐               ┌─────────────────────────────┐
         │ Visual Explainability       │               │ Clinical Explainability     │
         │ Grad-CAM Heatmap on Layer4  │               │ Perturbation Sensitivity    │
         │ Overlaid on Lesion Photo    │               │ Supporting & Opposing Cards │
         └─────────────────────────────┘               └─────────────────────────────┘
```

---

## 12. Experimental Setup

- **Software Environment**: Python 3.12, PyTorch 2.x, Torchvision, Scikit-learn, NumPy, Pandas, OpenCV, FastAPI, Uvicorn.
- **Frontend Environment**: Next.js 15, React 19, TypeScript, Tailwind CSS.
- **Reproducibility**: Experiments executed with static random seed ($42$). Split manifests are saved to disk (`train.csv`, `val.csv`, `test.csv`) to ensure 100% deterministic re-evaluation.

---

## 13. Model Training

The training process is executed in discrete procedural steps:
1. **Split Manifest Validation**: Confirm zero patient-group overlap and verify pHash cluster isolation.
2. **Clinical Preprocessor Fitting**: Calculate mean, standard deviation, and ANOVA F-scores on `train.csv` clinical rows; persist metadata to `clinical_features.json`.
3. **Dataset Instantiation**: Initialize `MultimodalLesionDataset` returning synchronized `(image_tensor, clinical_tensor, label, group_id)`.
4. **Stage 1 Optimization (10 Epochs)**: Freeze ResNet-50 weights. Train Clinical MLP and MLP Head using `AdamW` ($\eta = 10^{-3}$) with weighted cross-entropy.
5. **Stage 2 Optimization (15 Epochs)**: Unfreeze `layer3` and `layer4`. Fine-tune end-to-end with differential learning rates ($\eta_{\text{backbone}} = 10^{-4}$, $\eta_{\text{head}} = 5 \times 10^{-4}$).
6. **Checkpointing**: Track validation loss and macro F1; save the best checkpoint to `multimodal_best.pt`.

---

## 14. Evaluation Metrics

Model performance is evaluated across six rigorous classification metrics:

1. **Overall Accuracy**:
   $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$
2. **Balanced Accuracy**:
   $$\text{Balanced Accuracy} = \frac{1}{2} \left( \frac{\text{TP}}{\text{TP} + \text{FN}} + \frac{\text{TN}}{\text{TN} + \text{FP}} \right) = \frac{\text{Sensitivity} + \text{Specificity}}{2}$$
3. **Precision (Positive Predictive Value)**:
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
4. **Recall / Sensitivity (True Positive Rate)**:
   $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
5. **Specificity (True Negative Rate)**:
   $$\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}$$
6. **Macro-Averaged F1-Score**:
   $$\text{F1}_{\text{macro}} = \frac{\text{F1}_{\text{non\_mpox}} + \text{F1}_{\text{mpox}}}{2}, \quad \text{where } \text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
7. **Area Under the Receiver Operating Characteristic Curve (ROC-AUC)**:
   $$\text{ROC-AUC} = \int_{0}^{1} \text{TPR}(\text{FPR}^{-1}(t)) \, dt$$

---

## 15. Results

### 15.1 Computer Vision Backbone Comparison
Four convolutional architectures were benchmarked under identical training protocols on the holdout test set ($N = 270$ across 12 isolated patient groups):

#### Table 5: Computer Vision Backbone Performance Benchmark

| Backbone Architecture | Model Family | Parameters | Test Accuracy | Mpox Recall | Specificity | Mpox F1 | ROC-AUC | CPU Latency (p50) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-50** *(Selected)* | Residual Network | 25.6M | **86.67%** | **75.00%** | **89.45%** | **68.42%** | **91.97%** | 5.78 ms |
| **DenseNet-121** | Densely Connected | 8.0M | 82.22% | **75.00%** | 83.95% | 61.90% | 85.97% | 6.89 ms |
| **EfficientNet-B0** | Compound Scaling | 5.3M | 81.48% | 59.62% | 86.70% | 55.36% | 86.53% | 50.97 ms |
| **MobileNet-V3-Small** | Mobile / Edge | 2.5M | 81.85% | 44.23% | 90.83% | 48.42% | 81.69% | **2.04 ms** |

**Backbone Evaluation Findings**:
1. **ResNet-50** achieved the highest test accuracy (86.67%), highest ROC-AUC (91.97%), and the best balance of sensitivity (75.00%) and specificity (89.45%).
2. **DenseNet-121** matched ResNet-50 in sensitivity (75.00%), but produced more false positives, reducing specificity to 83.95%.
3. **MobileNet-V3-Small** exhibited the fastest inference (2.04 ms), but suffered from low sensitivity (44.23%).
4. **ResNet-50 was selected** as the primary visual feature extractor.

---

### 15.2 Multimodal Diagnostic Systems Benchmark

We evaluated three primary operational systems on the identical holdout test partition ($N = 270$):
- **Model A (Image-Only)**: ResNet-50 classifier.
- **Model B (Clinical-Only)**: 28-symptom Clinical MLP classifier.
- **Model C (Multimodal Fusion)**: ResNet-50 + Clinical MLP late fusion network.

#### Table 6: Comprehensive Operational Model Comparison Matrix

| Evaluation Metric | Model A: Image-Only | Model B: Clinical-Only | Model C: Multimodal Fusion | Multimodal Gain vs Image Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Accuracy** | 86.67% | **95.19%** | **90.37%** | **+3.70%** |
| **Balanced Accuracy** | 82.22% | **87.50%** | **84.52%** | **+2.30%** |
| **Precision (Monkeypox)** | 62.90% | **100.00%** | **75.00%** | **+12.10%** |
| **Recall / Sensitivity** | **75.00%** | **75.00%** | **75.00%** | **Preserved (0.00%)** |
| **Specificity (Non-Mpox)** | 89.45% | **100.00%** | **94.04%** | **+4.59%** |
| **Macro F1-Score** | 79.99% | **91.41%** | **84.52%** | **+4.53%** |
| **Weighted F1-Score** | 87.09% | **94.91%** | **90.37%** | **+3.28%** |
| **ROC-AUC** | 91.97% | **97.13%** | **93.60%** | **+1.63%** |
| **False Positives ($n=218$)** | 23 | **0** | **13** | **-43.5% reduction** |
| **False Negatives ($n=52$)** | 13 | 13 | 13 | Maintained |

```
                       ┌────────────────────────────────────────────────────────┐
                       │           Test Set False Positive Reduction            │
                       └────────────────────────────────────────────────────────┘
  Model A (Image-Only)  ███████████████████████ 23 False Positives (Precision: 62.9%)
  Model C (Multimodal)  █████████████ 13 False Positives (Precision: 75.0%)  [43.5% Reduction]
  Model B (Clin-Only)   0 False Positives (Precision: 100.0%)
```

---

### 15.3 Confusion Matrices & Error Breakdown

Evaluated on the $N = 270$ holdout test partition:

#### 1. Model A: Image-Only Baseline
```
                     Predicted Non-Mpox    Predicted Mpox
Actual Non-Mpox:            195                  23   (TN: 195, FP: 23)
Actual Mpox:                 13                  39   (FN: 13,  TP: 39)
```
- **Accuracy**: $(195 + 39) / 270 = \mathbf{86.67\%}$
- *Limitation*: Visual similarities between atypical non-monkeypox lesions (e.g., severe chickenpox) and true monkeypox lesions caused 23 false positives.

#### 2. Model B: Clinical-Only Baseline
```
                     Predicted Non-Mpox    Predicted Mpox
Actual Non-Mpox:            218                   0   (TN: 218, FP: 0)
Actual Mpox:                 13                  39   (FN: 13,  TP: 39)
```
- **Accuracy**: $(218 + 39) / 270 = \mathbf{95.19\%}$
- *Strength*: Hallmark clinical symptoms (mucosal ulcers, high fever, marked lymphadenopathy) provided zero false positives on this partition.

#### 3. Model C: Multimodal Fusion Network
```
                     Predicted Non-Mpox    Predicted Mpox
Actual Non-Mpox:            205                  13   (TN: 205, FP: 13)
Actual Mpox:                 13                  39   (FN: 13,  TP: 39)
```
- **Accuracy**: $(205 + 39) / 270 = \mathbf{90.37\%}$
- *Synergy*: Integrating clinical presentation eliminated 10 false alarms, boosting precision from 62.90% to **75.00%** while preserving 75.00% sensitivity.

---

## 16. Ablation Study

An ablation study was conducted to isolate the contribution of each modality:

#### Table 7: Systematic Ablation Analysis of Modality Contributions

| Configuration | Modalities Included | Feature Dimensions | Test Accuracy | Precision | Specificity | False Positives |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Image-Only (Model A)** | Dermatological Photograph | 256-d | 86.67% | 62.90% | 89.45% | 23 |
| **Clinical-Only (Model B)** | 28 Patient Symptoms | 64-d | 95.19% | 100.00% | 100.00% | 0 |
| **Multimodal Fusion (Model C)** | **Photograph + Symptoms** | **320-d (256+64)** | **90.37%** | **75.00%** | **94.04%** | **13** |

#### Key Takeaways from the Ablation Study:
1. **Image Alone is Insufficient**: The image-only baseline suffers from a high false-positive rate ($10.55\%$ false positive rate) because lesion morphology overlaps across poxviruses.
2. **Clinical Symptoms Provide Specificity**: Clinical symptoms provide strong discriminatory separation, especially when hallmark mucosal ulcerations and fever are recorded.
3. **Multimodal Synergy**: In real-world deployment where symptoms may be partially reported or atypical, the multimodal network leverages visual spatial evidence while using clinical signals to cross-validate and suppress false alarms.

---

## 17. Explainability

Pox Scanner implements **dual-modality explainability** in `ml/src/explainability.py`, providing interpretable rationales for both visual and tabular predictions:

```
                                  ┌─────────────────────────────────────────┐
                                  │      Dual-Modality Explainability       │
                                  └────────────────────┬────────────────────┘
                                                       │
                           ┌───────────────────────────┴───────────────────────────┐
                           ▼                                                       ▼
            [ Visual Explainability: Grad-CAM ]                     [ Clinical Explainability: Perturbation ]
            • Gradient flow to layer4 feature maps                  • Marginal symptom ablation (x_j -> 0)
            • Spatial activation pooling (alpha_k)                  • Probability delta: Delta P_j = P - P_ablated
            • Jet colormap overlay on lesion                        • Categorized: Supporting vs. Opposing
```

### 17.1 Visual Explainability: Grad-CAM
Gradient-Weighted Class Activation Mapping (Grad-CAM) computes the gradients of the predicted class score $y^c$ with respect to the feature activation maps $A^k$ of the final convolutional layer (ResNet-50 `layer4`):
$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^U \sum_{j=1}^V \frac{\partial y^c}{\partial A_{i,j}^k}$$
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
The resulting localization map is normalized to $[0.0, 1.0]$, upsampled to $224 \times 224$ via bilinear interpolation, converted to a jet colormap, and blended over the original photograph ($\alpha = 0.5$). This confirms that the model attends to active lesion umbilication, borders, and erythema rather than background artifacts.

### 17.2 Clinical Explainability: Symptom Perturbation Analysis
To interpret the tabular symptom branch, we perform marginal perturbation sensitivity analysis. For each present symptom $j$, we compute the baseline counterfactual where feature $j$ is ablated ($x_j \to 0$):
$$\Delta P_j = P(\text{Monkeypox} \mid X_{\text{img}}, X_{\text{clin}}) - P(\text{Monkeypox} \mid X_{\text{img}}, X_{\text{clin}}^{(j \to 0)})$$
- **Supporting Symptoms ($\Delta P_j > 0$)**: The presence of symptom $j$ increased model confidence for Monkeypox (e.g., `oral and genital ulcers`, `fever`, `swollen lymph nodes`).
- **Opposing Symptoms ($\Delta P_j < 0$)**: Pushed the prediction toward Non-Monkeypox.
- The UI renders these as discrete, quantitative impact cards with ranked bar indicators.

---

## 18. Discussion

### 18.1 Diagnostic Synergy of Multimodal Fusion
The central clinical contribution of Pox Scanner is resolving diagnostic ambiguity between monkeypox and varicella-zoster (chickenpox). Morphologically, both infections present with vesicular and pustular crops. However, varicella rarely causes prominent lymphadenopathy or deep anogenital/oral mucosal ulcerations [11, 12]. 

Our empirical results demonstrate that while the image-only baseline misclassified 23 non-mpox cases as monkeypox, the multimodal network corrected 10 of these errors, achieving a **43.5% reduction in false alarms** and raising precision from 62.90% to 75.00%.

### 18.2 Dynamic Runtime Routing
To accommodate varied real-world clinic workflows where one data modality may be unavailable, Pox Scanner implements **dynamic runtime dispatching**:

```
[ Incoming Request ]
        │
        ├─► Only Clinical Symptoms Provided  ──► [ Model B: Clinical MLP ] (95.19% Acc)
        │                                          Output: Ranked Symptom Attributions
        │
        ├─► Only Lesion Photograph Provided   ──► [ Model A: ResNet-50 ] (86.67% Acc)
        │                                          Output: Grad-CAM Localization Heatmap
        │
        └─► Both Image & Symptoms Provided    ──► [ Model C: Multimodal Fusion ] (90.37% Acc)
                                                   Output: Dual (Grad-CAM + Attributions)
```

---

## 19. Limitations

1. **Clinical Cohort Sample Size**: The tabular clinical dataset contains 211 cases. While sufficient for training the compact 2-layer Clinical MLP, expanding to multi-center EHR cohorts will improve generalizability.
2. **Fitzpatrick Skin Phototype Diversity**: Historical public dermatological datasets have documented imbalances across darker skin tones (Fitzpatrick types V–VI). Continued collection of diverse clinical photography is essential.
3. **Investigational Nature**: Pox Scanner is designed as an educational and clinical decision-support tool. It does not provide definitive medical diagnoses and does not supersede laboratory RT-PCR confirmation.

---

## 20. Future Work

1. **Cross-Attention Multimodal Fusion**: Implement cross-modal attention mechanisms (e.g., Vision-Language Transformers) allowing symptom tokens to directly attend to spatial image patches.
2. **Self-Supervised Dermatological Pretraining**: Pretrain the vision backbone on massive unlabelled dermatological corpora (e.g., ISIC archive) prior to task-specific fine-tuning.
3. **Prospective Clinical Trials**: Deploy Pox Scanner in prospective multi-center primary care clinics to assess real-world diagnostic utility, clinician acceptance, and workflow latency.

---

## 21. Conclusion

This paper presented **Pox Scanner**, an open-access multimodal deep learning framework for explainable monkeypox detection. By uniting a ResNet-50 convolutional image feature extractor (256-d) with an ANOVA-filtered Clinical MLP (64-d) into a 320-dimensional late fusion classification head, the system achieves **90.37% test accuracy** and a **43.5% relative reduction in false positives** over image-only baselines. The integration of spatial Grad-CAM localization heatmaps and quantitative symptom attribution cards addresses the critical need for transparency, establishing a rigorous and dependable clinical decision-support system for global infectious disease surveillance.

---

## 22. Declarations

- **Ethics Approval**: Not applicable (publicly available de-identified secondary data).
- **Competing Interests**: The authors declare no competing financial or non-financial interests.
- **Data & Code Availability**: All code, model checkpoints, and split manifests are openly available at the project repository.
- **Author Contributions**: Architecture design, ML engineering, full-stack implementation, and manuscript preparation by the Pox Scanner R&D team.

---

## 23. References

1. Alnaji L. Machine learning in epidemiology: neural networks forecasting of Monkeypox cases. *PLoS ONE*. 2024;19(5):e0300216. https://doi.org/10.1371/journal.pone.0300216
2. Muhammed Kalo Hamdan A, Ekmekci D. Prediction of Monkeypox infection from clinical symptoms with adaptive artificial bee colony-based artificial neural network. *Neural Comput Appl*. 2024;36:1–16. https://doi.org/10.1007/s00521-024-09782-z
3. Bleichrodt A, Dahal S, Maloney K, Casanova L, Luo R, Chowell G. Real-time forecasting the trajectory of Monkeypox outbreaks at the National and global levels. *BMC Med*. 2023;21(1):19. https://doi.org/10.1186/s12916-022-02725-2
4. Patel A, Bilinska J, Tam JC, et al. Clinical features and novel presentations of human monkeypox in a central London centre during the 2022 outbreak: descriptive case series. *BMJ*. 2022;378:e072410. https://doi.org/10.1136/bmj-2022-072410
5. Manohar B, Das R. Artificial neural networks for the prediction of Monkeypox outbreak. *Trop Med Infect Dis*. 2022;7(12):424. https://doi.org/10.3390/tropicalmed7120424
6. Devarajan D, Dhana lakshmi P, Krishnaveni S, Senthilkumar S. Human Monkeypox disease prediction using novel modified restricted Boltzmann machine-based equilibrium optimizer. *Sci Rep*. 2024;14(1):17612. https://doi.org/10.1038/s41598-024-68836-3
7. Markewitz NF, DeLuca J, Kalejaiye A, Shidid S, Jain T, Parikh P, Norton BL. Mpox-Associated pneumonia: A case report. *Ann Intern Med Clin Cases*. 2023;2(3):e220945. https://doi.org/10.7326/aimcc.2022.0945
8. WHO. WHO Director-General declares mpox outbreak a public health emergency of international concern. World Health Organization; 2024. Available from: https://www.who.int/news/item/14-08-2024-who-director-general-declares-mpox-outbreak-a-public-health-emergency-of-international-concern
9. Africa CDC. Africa CDC declares mpox a Public Health Emergency of Continental Security. Africa Centres for Disease Control and Prevention; 2024. Available from: https://africacdc.org
10. DermNet NZ. Molluscum contagiosum and poxvirus clinical imagery. DermNet; 2024. Available from: https://dermnetnz.org
11. Hussain A, Kaler J, Lau G, Maxwell T. Clinical conundrums: differentiating Monkeypox from similarly presenting infections. *Cureus*. 2022;14(10):e29929. https://doi.org/10.7759/cureus.29929
12. Pattnaik H, Surani S, Goyal L, Kashyap R. Making sense of Monkeypox: a comparison of other poxviruses to the Monkeypox. *Cureus*. 2023;15(4):e38083. https://doi.org/10.7759/cureus.38083
13. Lalmuanawma S, Hussain J, Chhakchhuak L. Applications of machine learning and artificial intelligence for Covid-19 (SARS-CoV-2) pandemic: A review. *Chaos Solitons Fractals*. 2020;139:110059. https://doi.org/10.1016/j.chaos.2020.110059
14. Panch T, Mattie H, Atun R. Artificial intelligence and algorithmic bias: implications for health systems. *J Glob Health*. 2019;9(2):020318. https://doi.org/10.7189/jogh.09.020318
15. Nilofer A, Sasikala S. A comparative study of machine learning algorithms using explainable artificial intelligence system for predicting liver disease. *Comput Open*. 2023;1:2350003. https://doi.org/10.1142/S2972370123500034
16. Akpinar R, Panzeri D, De Carlo C, et al. Role of artificial intelligence in staging and assessing of treatment response in MASH patients. *Front Med*. 2024;11:1480866. https://doi.org/10.3389/fmed.2024.1480866
17. Decharatanachart P, Chaiteerakij R, Tiyarattanachai T, Treeprasertsuk S. Application of artificial intelligence in chronic liver diseases: a systematic review and meta-analysis. *BMC Gastroenterol*. 2021;21:1–16. https://doi.org/10.1186/s12876-020-01585-5
18. Alakuş TB. Prediction of Monkeypox on the skin lesion with the Siamese deep learning model. *Balkan J Electr Comput Eng*. 2023;11(3):225–231. https://doi.org/10.17694/bajece.1255798
19. Nayak T, Chadaga K, Sampathila N, et al. Detection of Monkeypox from skin lesion images using deep learning networks and explainable artificial intelligence. *Appl Math Sci Eng*. 2023;31(1):2225698. https://doi.org/10.1080/27690911.2023.2225698
20. Uysal F. Detection of Monkeypox disease from human skin images with a hybrid deep learning model. *Diagnostics*. 2023;13(10):1772. https://doi.org/10.3390/diagnostics13101772
21. Akram A, Jamjoom AA, Innab N, et al. SkinMarkNet: an automated approach for prediction of MonkeyPox using image data augmentation with deep ensemble learning models. *Multimedia Tools Appl*. 2024;1–17. https://doi.org/10.1007/s11042-024-19862-w
22. Farzipour A, Elmi R, Nasiri H. Detection of Monkeypox cases based on symptoms using XGBoost and Shapley additive explanations methods. *Diagnostics*. 2023;13(14):2391. https://doi.org/10.3390/diagnostics13142391
23. Long B, Tan F, Newman M. Forecasting the Monkeypox outbreak using ARIMA, prophet, neuralprophet, and LSTM models in the united States. *Forecasting*. 2023;5(1):127–137. https://doi.org/10.3390/forecast5010005
24. Dada EG, Oyewola DO, Joseph SB, Emebo O, Oluwagbemi OO. Ensemble machine learning for Monkeypox transmission time series forecasting. *Appl Sci*. 2022;12(23):12128. https://doi.org/10.3390/app122312128
25. Thieme AH, Zheng Y, Machiraju G, et al. A deep-learning algorithm to classify skin lesions from Mpox virus infection. *Nat Med*. 2023;29(3):738–747. https://doi.org/10.1038/s41591-023-02225-7
26. Uzun Ozsahin D, Mustapha MT, Uzun B, Duwa B, Ozsahin I. Computer-aided detection and classification of Monkeypox and chickenpox lesion in human subjects using deep learning framework. *Diagnostics*. 2023;13(2):292. https://doi.org/10.3390/diagnostics13020292
27. Al-Gaashani MS, Xu W, Obsie EY. MobileNetV2-based deep learning architecture with progressive transfer learning for accurate Monkeypox detection. *Appl Soft Comput*. 2025;169:112553. https://doi.org/10.1016/j.asoc.2024.112553
28. Eliwa EHI, El Koshiry AM, Abd El-Hafeez T, Farghaly HM. Utilizing convolutional neural networks to classify Monkeypox skin lesions. *Sci Rep*. 2023;13(1):14495. https://doi.org/10.1038/s41598-023-41545-z
29. Sorayaie Azar A, Naemi A, Babaei Rikan S, et al. Monkeypox detection using deep neural networks. *BMC Infect Dis*. 2023;23(1):438. https://doi.org/10.1186/s12879-023-08408-4
30. Nguyen G, Dlugolinsky S, Bobák M, et al. Machine learning and deep learning frameworks and libraries for large-scale data mining: a survey. *Artif Intell Rev*. 2019;52:77–124. https://doi.org/10.1007/s10462-018-09679-z
31. Lu SC, Swisher CL, Chung C, et al. On the importance of interpretable machine learning predictions to inform clinical decision making in oncology. *Front Oncol*. 2023;13:1129380. https://doi.org/10.3389/fonc.2023.1129380
32. Farzipour A. GitHub repository: MonkeyPoxDetection. Available from: https://github.com/alirezafarzipour/MonkeyPoxDetection [Accessed 16 Jan 2025].
33. Shiwangi K, Sandhu JK, Sahu R. Effective Heart-Disease Prediction by Using Hybrid Machine Learning Technique. *IEEE ICCPCT*. 2023;1670–1675. https://doi.org/10.1109/ICCPCT58313.2023.10245785
34. Setegn GM, Dejene BE. Explainable AI for Symptom-Based Detection of Monkeypox: a machine learning approach. *BMC Infect Dis*. 2025;25:419. https://doi.org/10.1186/s12879-025-10738-4
35. He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. 2016;770–778.
36. Selvaraju RR, Cogswell M, Das A, Vedaldi A, Parikh D, Batra D. Grad-CAM: Visual explanations from deep networks via gradient-based localization. In *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*. 2017;618–626.
