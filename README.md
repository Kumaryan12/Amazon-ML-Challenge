# Amazon_ML_Challenge_2025

## Overview
This repository contains the complete implementation for the **Amazon ML Challenge 2025**.  
The project predicts product prices using multimodal information (text, image, and metadata).  
All experiments, preprocessing, and model training have been executed inside the provided Jupyter notebooks.

---

## Repository Structure
```bash
AMAZON_ML_CHALLENGE/
│
├── .gitignore
├── LICENSE
├── README.md
│
├── Notebook_Till_Checkpoint.ipynb        # Full pipeline till checkpoint creation
├── Notebook_From_CheckPoint.ipynb        # Resumed notebook using pre-saved checkpoint
│
├── checkpoint_bundle.tar.gz              # Contains saved model states, embeddings, and metadata
│
└── test_predictions_seg_cd_blend_46.8361.csv   # Final output predictions (SMAPE ~46.83%)
```
## Project Description
# Task
Predict realistic selling prices for products listed on an e-commerce platform using multimodal data.
The evaluation metric is SMAPE (Symmetric Mean Absolute Percentage Error).

# Input Data
Each product includes:

Title and Description (text)

Product Category, Brand, and Segment (categorical)

Image Links (visual input)

Numerical metadata (ratings, discount, etc.)

# Output
Predicted numeric price for each test entry.

Implementation Workflow
All implementation steps are consolidated inside two Jupyter notebooks:

1. Notebook_Till_Checkpoint.ipynb
This notebook executes:

Data preprocessing and cleaning

Feature extraction:

Text embeddings using e5-large-v2

Image embeddings using OpenCLIP (ViT-L/14)

Metadata encoding and normalization

Base model training (LightGBM, XGBoost, Ridge, FAISS-based kNN)

Checkpoint creation: trained models and embeddings saved to checkpoint_bundle.tar.gz

2. Notebook_From_CheckPoint.ipynb
This notebook resumes from the saved checkpoint and performs:

Model restoration and validation

Meta-model (residual booster + isotonic regression)

Segment-wise blending and calibration

Generation of final predictions → test_predictions_seg_cd_blend_46.8361.csv

# Checkpoint Description
The file checkpoint_bundle.tar.gz contains:

``` pgsql
Copy code
/checkpoint/
├── model_weights/          # XGBoost, LightGBM, Ridge, Meta-Model
├── text_embeddings/        # TF-IDF + e5-large-v2 sentence embeddings
├── image_embeddings/       # OpenCLIP ViT-L/14 features
├── scaler_params/          # Feature normalizers and encoders
└── metadata.pkl            # Dataset configuration and fold info
```
To reuse:

```bash
Copy code
# extract
tar -xvzf checkpoint_bundle.tar.gz
# place extracted 'checkpoint/' directory at project root
```
## Execution Instructions
# Run Entire Pipeline from Start
```bash
Copy code
jupyter notebook Notebook_Till_Checkpoint.ipynb
```
## Resume from Saved Checkpoint
```bash
Copy code
jupyter notebook Notebook_From_CheckPoint.ipynb
```
All results (predictions, validation scores, and model summaries) are generated within the notebooks.

Results Summary
Model Type	Description	SMAPE (CV)
Text-only (TF-IDF + Ridge)	Text baseline	63.4%
Multimodal LGBM	Text + Meta	49.5%
Per-Segment XGB	Text + Image + Meta	48.5%
Residual Superblend	Meta-ensemble	47.2%
Seg-CD Blend (Final)	Calibrated final blend	46.83%

Dependencies
```bash
Copy code
python==3.10
numpy>=1.26
pandas>=2.0
scikit-learn>=1.5
xgboost>=2.0
lightgbm>=4.0
faiss-cpu
sentence-transformers
open_clip_torch
matplotlib
seaborn
```
To install:

```bash
Copy code
pip install -r requirements.txt
```
Reproducibility Notes
Random seeds fixed for all frameworks

Pretrained embeddings reused via checkpoint to save time

No external data used beyond the official challenge dataset

License
This project is released under the MIT License (see LICENSE file).

Author
Aryan Kumar, NIT Goa
Amazon ML Challenge 2025 — Multimodal Price Prediction Solution