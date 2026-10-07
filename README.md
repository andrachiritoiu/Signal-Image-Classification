# Radio Signal Image Classification

The project compares classical machine learning baselines, hand-crafted feature pipelines, gradient boosting and **CNNs trained entirely from scratch**. The best solution is a weighted **ensemble of two 2D-CNNs (asymmetric pooling) and one 1D-Profile-CNN**, reaching **0.7571 validation accuracy**.

> **Constraints respected:** no external data, no pretrained models, no transfer learning.

> **Note:** all accuracies in this document are computed locally on a stratified validation split. Scores obtained on Kaggle were generally higher, by up to ~0.03.

## Table of Contents

1. [Problem & Dataset](#problem--dataset)
2. [Results at a Glance](#results-at-a-glance)
3. [General Methodology](#general-methodology)
4. [Models in Detail](#models-in-detail)
   - [Classical baselines (Models 1–3)](#classical-baselines-models-13)
   - [Hand-crafted features (Models 4–6)](#hand-crafted-features-models-46)
   - [XGBoost (Models 7–8)](#xgboost-models-78)
   - [CNNs from scratch (Models 9–10)](#cnns-from-scratch-models-910)
   - [Advanced CNNs & final ensemble](#advanced-cnns--final-ensemble)
5. [Final Ensemble](#final-ensemble)
6. [Key Takeaways](#key-takeaways)
7. [Requirements](#requirements)
8. [Reproducibility](#reproducibility)
9. [Author](#author)

---

## Problem & Dataset

The task is to classify images of radio signals (`.png`) into **one of 5 classes**. The files `train.csv` and `test.csv` contain the image names and, for the training set, the labels.

| Set | Images |
|---|---|
| Train | 15,500 |
| Test | 5,500 |
| Classes | 5 |

**Class distribution (train):**

| Original class | Internal label | Images |
|---|---|---|
| 1 | 0 | 3,500 |
| 2 | 1 | 3,000 |
| 3 | 2 | 3,000 |
| 4 | 3 | 3,000 |
| 5 | 4 | 3,000 |

**Local split:** stratified **80% / 20%**

| Subset | Images |
|---|---|
| Local train | 12,400 |
| Validation | 3,100 |

**Class weights** (used in the loss function of all CNN models, because class 1 is over-represented):

| Class | Weight |
|---|---|
| 1 | 0.8857 |
| 2 | 1.0333 |
| 3 | 1.0333 |
| 4 | 1.0333 |
| 5 | 1.0333 |

---

## Results at a Glance

| # | Method | Validation accuracy |
|---|---|---|
| 1 | KNN (64×64, L1, k=5) | 0.2223 |
| 2 | Linear SVM on raw pixels | 0.2048 |
| 3 | Extra Trees on raw pixels | 0.2684 |
| 4 | HOG + Linear SVM | 0.3026 |
| 5 | HOG + Global + Hough + Linear SVM | 0.3252 |
| 6 | HOG + Global + Hough + Softmax | 0.3342 |
| 7 | XGBoost + extended feature extraction | **0.4206** *(best classical)* |
| 8 | XGBoost + PCA (200 components) | 0.3577 |
| 9 | ScratchSignalCNN (RGB) | 0.6881 |
| 10 | DeepSignalCNN (RGB) | 0.6987 |
| 10 | DeepSignalCNN RGB + EDGE + FREQ ensemble | 0.6997 |
| 11 | DeepSignalCNN GREEN + Sliding Window 1×7 | 0.7139 |
| 12 | 2D-CNN with asymmetric pooling | 0.7558 |
| 12 | 2D-CNN + 1D-Profile-CNN ensemble | 0.7561 |
| 13 | **Final ensemble** (2D old + 2D extra seed + 1D profile) | **0.7571** |

---

## General Methodology

Complexity was increased progressively, starting from simple models used to validate the full pipeline:

1. Read images from `.png` files
2. Stratified train/validation split
3. Model-specific preprocessing
4. Training
5. Hyperparameter selection on validation
6. Confusion-matrix analysis
7. Test predictions and submission file generation

Model families tested:

- **Raw-pixel models:** KNN, Linear SVM, Extra Trees
- **Hand-crafted features:** HOG, LBP, intensity histograms, row/column projections, FFT, connected components, Hough
- **Linear models on features:** Linear SVM, Softmax (Logistic Regression)
- **Gradient boosting:** XGBoost (with and without PCA)
- **CNNs trained from scratch**
- **Ensembles of CNNs** with temperature scaling and test-time augmentation (TTA)

---

## Models in Detail

### Classical baselines (Models 1–3)

#### Model 1 — K-Nearest Neighbors
- Preprocessing: PIL → grayscale → resize **64×64** → divide by 255 → `flatten()` (4096 features)
- Tested L1 and L2 distances with k ∈ {1, 3, 5, 7, 9}
- **Best:** k = 5, L1 → **0.2223**
- Weak because it compares pixels directly and learns no abstract visual structure.

#### Model 2 — Linear SVM on raw pixels
- Same preprocessing as KNN + `StandardScaler` (fit on local train only)
- Tuned `C` ∈ {0.001, 0.01, 0.1, 1, 10}
- **Best:** C = 0.001 → **0.2048** (did not beat KNN)

#### Model 3 — Extra Trees
- Tuned `n_estimators` ∈ {100, 300, 500} and `max_depth` ∈ {None, 20, 40}
- **Best:** `n_estimators=100`, `max_depth=40` → **0.2684**

### Hand-crafted features (Models 4–6)

#### Model 4 — HOG + Linear SVM
Pipeline: `Image → grayscale → 64×64 → HOG → StandardScaler → LinearSVC`

| HOG parameter | Value |
|---|---|
| `orientations` | 12 |
| `pixels_per_cell` | (8, 8) |
| `cells_per_block` | (2, 2) |
| `block_norm` | L2-Hys |

- **Best:** C = 0.0002 → **0.3026** (tied with C = 0.0003, but with slightly stronger regularization)

#### Model 5 — HOG + Global Features + Hough + Linear SVM
Added features suited to visual radio signals (**2,449 features per image**):
- Global pixel-intensity statistics
- Distribution of bright pixels across rows and columns
- Number and area of connected components
- Hough-based line-detection features

Tuned `C` (1e-6 … 1.2e-5) and `class_weight` ∈ {None, balanced}.
**Best:** C = 7·10⁻⁶, no class weight → **0.3252**

#### Model 6 — HOG + Global + Hough + Softmax
- `LogisticRegression` (multi-class linear model that optimizes a different loss than SVM)
- **Best:** C = 0.0001 → **0.3342**

### XGBoost (Models 7–8)

#### Model 7 — XGBoost + extended feature extraction
Feature set: **HOG, LBP, intensity histogram, row/column projections, FFT, connected components, Hough features**. No `StandardScaler` is applied (tree models are not scale-sensitive).

| n_estimators | max_depth | learning_rate | subsample | colsample | reg_lambda | Accuracy |
|---|---|---|---|---|---|---|
| 300 | 2 | 0.03 | 0.9 | 0.9 | 3 | 0.3758 |
| 300 | 3 | 0.03 | 0.9 | 0.9 | 3 | 0.3968 |
| 500 | 3 | 0.05 | 0.9 | 0.8 | 3 | 0.4094 |
| 500 | 4 | 0.03 | 0.8 | 0.8 | 5 | **0.4206** |

#### Model 8 — XGBoost + PCA
Pipeline: `Image → feature extraction → StandardScaler → PCA → XGBoost`

| PCA | Components | Explained variance | Best val. acc. |
|---|---|---|---|
| 50 | 50 | 0.5819 | 0.3513 |
| 100 | 100 | 0.6547 | 0.3458 |
| 200 | 200 | 0.7363 | **0.3577** |
| 0.95 | 798 | 0.9500 | 0.3355 |

PCA **hurt** performance: it discarded information that was useful for separating the classes.

### CNNs from scratch (Models 9–10)

#### Model 9 — ScratchSignalCNN (RGB + EDGE)
- Input: **128×128**
- **RGB** variant: original image
- **EDGE** variant (3 channels): grayscale, Sobel magnitude, absolute Laplacian
- Augmentations: `RandomHorizontalFlip`, `RandomAffine`, `ColorJitter`, `RandomErasing`, `Normalize`
- Architecture: Conv-BN-SiLU blocks, residual blocks, `Dropout2d`, Adaptive Average Pooling, fully-connected classifier

| Setting | Value |
|---|---|
| Loss | CrossEntropy + label smoothing |
| Optimizer | AdamW |
| Learning rate | 3·10⁻⁴ |
| Weight decay | 1·10⁻⁴ |
| Scheduler | CosineAnnealingLR |
| Batch size | 64 |
| Max epochs / patience | 70 / 14 |

| Model | Best epoch | Val. acc. |
|---|---|---|
| RGB | 60 | 0.6881 |
| EDGE | 52 | 0.6713 |
| RGB + EDGE ensemble | – | 0.6881 (best weights: 1.00 RGB / 0.00 EDGE) |

#### Model 10 — DeepSignalCNN (SE + ASPP; RGB + EDGE + FREQ)
- Conv blocks, BatchNorm, SiLU, residual connections, **Squeeze-and-Excitation** and **ASPP**
- **11,751,333** trainable parameters per model
- Three input representations: **RGB**, **EDGE**, **FREQ**
- FREQ channels (2D Fourier transform of the grayscale image): log-magnitude for low frequencies, log-magnitude for high frequencies, cosine of the phase

| Setting | Value |
|---|---|
| Image size | 128×128 |
| Batch size | 48 |
| Max epochs / patience | 80 / 18 |
| Learning rate | 2·10⁻⁴ |
| Weight decay | 2·10⁻⁴ |
| Label smoothing | 0.08 |
| Hardware | NVIDIA A100-SXM4-80GB |

| Model | Best epoch | Val. acc. |
|---|---|---|
| RGB | 77 | 0.6987 |
| EDGE | 57 | 0.6800 |
| FREQ | 26 | 0.2829 |
| **RGB + EDGE + FREQ ensemble** | – | **0.6997** |

Temperature scaling values: RGB 1.0504, EDGE 0.8763, FREQ 1.4859. Best ensemble weights:

```
p = 0.85 · p_RGB + 0.05 · p_EDGE + 0.10 · p_FREQ
```

### Advanced CNNs & final ensemble

All models in this section were trained from scratch, without external data, pretrained models or transfer learning. The goal was to adapt the architecture to the structure of radio-signal images, which contain lines, bright bands and different distributions along the vertical and horizontal axes.

#### GREEN channel + Sliding Window 1×7
Three input channels: the original **green channel**, the **local mean over a 1×7 window**, and the **difference** between the green channel and its local mean. Trained for 90 epochs (initial lr 2·10⁻⁴, batch size 48, patience 20). Best result: **0.7139**.

| Epoch | Val. acc. |
|---|---|
| 10 | 0.4687 |
| 20 | 0.6506 |
| 30 | 0.6816 |
| 49 | 0.7052 |
| 75 | 0.7106 |
| 90 | 0.7139 |

#### 2D-CNN with Asymmetric Pooling + 1D-Profile-CNN
- **2D-CNN with asymmetric pooling** (12,044,069 params) preserves the directional structure of the signal better than symmetric pooling → **0.7558**
- **1D-Profile-CNN** (5,707,077 params) works on profiles extracted from the image (sum / mean of pixels over rows and columns). It is weak on its own, but contributes complementary information in an ensemble.

| Model | Params | Val. acc. |
|---|---|---|
| 2D-CNN asymmetric pool | 12,044,069 | 0.7558 |
| 1D-Profile-CNN | 5,707,077 | 0.3255 |
| 2D-CNN + 1D-Profile-CNN ensemble (90% / 10%) | – | 0.7561 |

The confusion matrix showed that the model recognizes class 0 very well but still confuses the neighbouring classes 3 and 4.

---

## Final Ensemble

The final submission keeps the main 2D-CNN and adds a second 2D-CNN trained with a different seed (`SEED_EXTRA=2026`). The 1D-Profile-CNN is kept as the third component for complementary information. Probabilities are calibrated with **temperature scaling** before being combined by weighted averaging.

| Model | Individual val. acc. | Temperature | Final weight |
|---|---|---|---|
| 2D-CNN old | 0.7542 | 1.1755 | 0.65 |
| 2D-CNN extra (seed 2026) | 0.7435 | 1.2004 | 0.10 |
| 1D-Profile-CNN | 0.2323 | 1.8802 | 0.25 |

> The individual accuracy of the 1D-Profile-CNN differs between runs (0.3255 in the intermediate ensemble above, 0.2323 for the model used in the final ensemble).

```
p_final = 0.65 · p_2D-old + 0.10 · p_2D-extra + 0.25 · p_1D-profile
```

**Final validation accuracy: 0.7571** — the best local score obtained in the project.

For the test set, **8 TTA predictions** were used for each component. The final file is `submission_LAST_BEST.csv`.

### Confusion matrix (rows = true class, columns = predicted class)

| True \ Pred | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| **0** | 661 | 23 | 10 | 5 | 1 |
| **1** | 118 | 455 | 15 | 8 | 4 |
| **2** | 54 | 76 | 446 | 16 | 8 |
| **3** | 35 | 28 | 107 | 393 | 37 |
| **4** | 21 | 24 | 50 | 113 | 392 |

### Classification report

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| 0 | 0.7435 | 0.9443 | 0.8320 | 700 |
| 1 | 0.7508 | 0.7583 | 0.7546 | 600 |
| 2 | 0.7102 | 0.7433 | 0.7264 | 600 |
| 3 | 0.7346 | 0.6550 | 0.6925 | 600 |
| 4 | 0.8869 | 0.6533 | 0.7524 | 600 |
| **Accuracy** | | | **0.7571** | 3100 |
| Macro avg | 0.7652 | 0.7509 | 0.7516 | 3100 |
| Weighted avg | 0.7645 | 0.7571 | 0.7542 | 3100 |

**Main confusions:** class 3 predicted as 2 (107 cases), class 4 predicted as 3 (113 cases) and class 2 predicted as 1 (76 cases). These errors are explained by the visual similarity between consecutive classes.

---

## Key Takeaways

- **Raw pixels are insufficient** — KNN, SVM and Extra Trees stay at 0.20–0.27.
- **Hand-crafted features help** — HOG, LBP, FFT, Hough, etc. lift accuracy to ~0.42 with XGBoost, the best classical method.
- **PCA did not help** XGBoost; it removed discriminative information.
- **CNNs trained from scratch gave the biggest jump** (from ~0.42 to ~0.70).
- **Domain-adapted architecture mattered most** — asymmetric pooling brought the largest single gain (0.7139 → 0.7558).
- **Ensembling diverse models** (different seed, 1D profile model, temperature scaling, TTA) added a final small boost to 0.7571.
- The FREQ (Fourier) representation was weak on its own (0.2829) and contributed little.

---

## Requirements

Main libraries used (versions not pinned in the report):

```
numpy
pandas
pillow
scikit-learn
scikit-image
opencv-python
xgboost
torch
torchvision
matplotlib
```

A CUDA-capable GPU is strongly recommended for the CNN models (the deep models were trained on an NVIDIA A100 80GB).

---

## Reproducibility

- Stratified 80% / 20% train/validation split
- `StandardScaler` and PCA fit **only** on the local train subset
- Hyperparameters selected on validation accuracy
- Second 2D-CNN trained with `SEED_EXTRA = 2026`
- Class-weighted loss, label smoothing, early stopping with patience, cosine annealing learning-rate schedule
- Test predictions use 8× TTA per ensemble component


