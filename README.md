
# Jaguar Re-Identification using Deep Metric Learning

This repository contains a complete **image-based re-identification (ReID) pipeline** built for the Kaggle *Jaguar Re-ID* challenge.  
The goal is to determine whether two images belong to the **same individual jaguar** by learning a robust embedding space and computing similarity scores.

This project follows **industry-standard computer vision and metric learning practices** and is designed to be reproducible, explainable, and extensible.

---

##  Problem Statement

You are given:
- A **training dataset** of jaguar images with identity labels
- A **test dataset** consisting of `(query_image, gallery_image)` pairs

Your task:
- Predict a **similarity score in the range `[0, 1]`**
- Higher score → higher likelihood both images depict the same jaguar

This is **not a classification problem**, but a **metric learning problem**.

---

##  Core Idea

Instead of predicting class labels, the model learns an **embedding representation** such that:

- Images of the **same jaguar** are close in embedding space
- Images of **different jaguars** are far apart

Similarity between two images is computed using **cosine similarity**.

---

##  Architecture Overview

### Backbone Network
- **ConvNeXt** (from `timm`)
- Pretrained on ImageNet
- Used as a **feature extractor** (classification head removed)

### Embedding Head
- Linear projection to 512 dimensions
- Batch Normalization
- L2 normalization

### Output
- Normalized embedding vector per image

---

##  Loss & Training Strategy

### Loss Function
- `CrossEntropyLoss`
- Optional **label smoothing** (e.g. `0.05`)

### Batch Sampling
- **PK Sampler**
  - P identities per batch
  - K images per identity
- Essential for metric learning
- Prevents identity imbalance

---

##  Evaluation Metric

### Identity-Balanced Mean Average Precision (mAP)

- Computes mAP **per identity**
- Averages across identities
- Prevents bias from identities with many images

Typical validation performance:
- **~0.79 – 0.82 identity-balanced mAP**

---

##  Similarity Computation

### Step 1 — Embedding Extraction
Each image is mapped to a normalized vector:

embedding ∈ ℝ⁵¹² , ||embedding||₂ = 1


### Step 2 — Cosine Similarity
cos_sim = dot(embedding_query, embedding_gallery)

### Step 3 — Kaggle-Compatible Scaling
Since Range of cosine similarity is `[-1, 1]`, Kaggle requires similarity values in `[0, 1]`:
similarity = (cos_sim + 1) / 2
