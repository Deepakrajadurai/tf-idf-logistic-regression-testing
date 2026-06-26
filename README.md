# 🤖 German AI Text Detector (TF-IDF + Logistic Regression)

An interpretable, lightweight, and ultra-fast AI text detection system for the German language, built using TF-IDF feature extraction and a Logistic Regression classifier. 

This pipeline was developed as a robust, explainable alternative to deep learning models (such as `gbert-large`) to investigate overfitting, dataset quality, and lexical shortcuts.

> **Overall Test Accuracy: 100.00% | Macro F1: 100.00%**  
> **Average CPU Inference Latency: 0.028 ms per text**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Performance & Calibration](#performance--calibration)
- [Top Stylistic Markers (Interpretability)](#top-stylistic-markers-interpretability)
- [The Generalization Challenge (Crucial Insight)](#the-generalization-challenge-crucial-insight)
- [License](#license)

---

## Overview

This repository implements a binary classifier that distinguishes **human-written** from **AI-generated** German text. The pipeline handles data quality diagnostics, de-duplication, and strict length-matching to ensure the classifier does not exploit text length as a classification shortcut.

Rather than relying on a heavy deep learning stack, it leverages a highly optimized **TF-IDF + Logistic Regression** pipeline, allowing it to run instantly on any CPU.

---

## Key Features

- 🎯 **100.00% Accuracy** on balanced, held-out test data (50k rows).
- ⚡ **Ultra-low latency** (~0.028 ms per text) on CPU—over 500x faster than BERT.
- 📏 **Quantile-based length balancing** to eliminate length-based shortcuts.
- 🔍 **Explainable & Interpretable**: Coefficients map directly back to German words/phrases.
- ⚙️ **Zero deep learning dependencies** (no PyTorch, TensorFlow, or CUDA required for core execution).

---

## Project Structure

```
├── requirements_tfidf.txt       # Lightweight package requirements
├── prepare_dataset_tfidf.py     # Step 1: Preprocessing, deduplication, length-balancing, & splits
├── train_tfidf.py               # Step 2: TF-IDF vectorization & Logistic Regression training/tuning
├── evaluate_tfidf.py            # Step 3: Evaluation, calibration analysis, & feature extraction
├── test_generalization_tfidf.py # Step 4: Generalization test on unseen real-world & synthetic AI styles
├── results/
│   ├── tfidf_evaluation_report.md  # Detailed evaluation report
│   ├── tfidf_feature_importance.csv# Coefficient weights for all vocabulary terms
│   └── tfidf_generalization_report.txt # Generalization metrics on out-of-distribution texts
└── models/
    └── tfidf_logreg/            # Saved vectorizer and classifier joblib binaries
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- ~500 MB free disk space (to store the raw dataset in `Data/`)

### 1. Clone the Repository

```bash
git clone https://github.com/Deepakrajadurai/tf-idf-logistic-regression-testing.git
cd tf-idf-logistic-regression-testing
```

### 2. Set Up Virtual Environment

```bash
python -m venv venv
# Activate on Windows:
venv\Scripts\activate
# Activate on Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements_tfidf.txt
```

---

## Pipeline Walkthrough

Execute the full pipeline step-by-step:

### Step 1: Data Preparation & Length Balancing
Clean, de-duplicate, and split the raw Bundestag speeches and AI sentences into length-balanced splits (80/10/10):
```bash
python prepare_dataset_tfidf.py
```
*Outputs: `Data/train_tfidf.csv`, `Data/val_tfidf.csv`, `Data/test_tfidf.csv`*

### Step 2: Train & Tune Classifier
Train both word-level and char-level TF-IDF pipelines, tune the regularization strength `C` on the validation set, and save the best model:
```bash
python train_tfidf.py
```
*Outputs: `models/tfidf_logreg/` containing saved models.*

### Step 3: Evaluate & Extract Interpretability
Evaluate on the holdout test split, compute calibration metrics, and export feature coefficients:
```bash
python evaluate_tfidf.py
```
*Outputs: `results/tfidf_evaluation_report.md` and `results/tfidf_feature_importance.csv`*

### Step 4: Generalization Suite
Check how well the model detects new AI text styles rewritten by a local LLM and real-world German news/legal articles:
```bash
python test_generalization_tfidf.py
```
*Outputs: `results/tfidf_generalization_report.txt`*

---

## Performance & Calibration

### Holdout Test Split (50,000 balanced rows)

| Model Setup | Accuracy | Macro F1 | ROC-AUC | ECE (Calibration) | Latency (per text) |
|---|---|---|---|---|---|
| **TF-IDF + LogReg** | **100.00%** | **100.00%** | **1.0000** | **0.0124** | **0.028 ms** (CPU) |
| GBERT-Large (Original) | 58.70% | 57.38% | 0.6718 | 0.4086 | ~15.0 ms (GPU) |
| GBERT-Large (Rotation 2)| 94.67% | 90.69% | 0.9932 | 0.0398 | ~15.0 ms (GPU) |

---

## Top Stylistic Markers (Interpretability)

Below are the top 10 features (n-grams) representing the strongest human-like and AI-like text markers in our dataset:

### 🤖 AI-Generated Indicators (Positive Coefficients)
1. `direkt` (14.19)
2. `bezüglich` (9.56)
3. `von` (8.50)
4. `unter` (8.48)
5. `abs` (6.22)
6. `drucksache` (5.94)
7. `az` (5.49)
8. `aufgrund der` (5.41)
9. `aktuellen` (5.35)
10. `aufgrund` (5.31)

### 👤 Human-Written Indicators (Negative Coefficients)
1. `auch` (-4.62)
2. `ist` (-3.70)
3. `es` (-3.54)
4. `sie` (-3.44)
5. `ich` (-2.98)
6. `haben` (-2.81)
7. `aber` (-2.46)
8. `das` (-2.26)
9. `nicht` (-2.23)
10. `und` (-2.17)

---

## The Generalization Challenge (Crucial Insight)

While the model scores a perfect **100.00%** on the in-distribution test set, our **Generalization Test** revealed a critical limitation:

* **Human Generalization**: **100% Accuracy** on unseen Bundestag speeches, news/wiki articles, and constitutional legal texts.
* **AI Generalization**: **0% Accuracy** on unseen AI text styles rewritten in ChatGPT, Claude, Gemini, or Qwen styles.

### Why does this happen?
Because of the interpretability of Logistic Regression, we can see that the model heavily relied on synthetic template markers (like `drucksache`, `az`, `abs`, and `aufgrund der aktuellen lage`) present in the training AI dataset. When evaluated on natural AI text that does not contain these specific template keywords, the model finds no positive markers and classifies 100% of the AI instances as human.

This highlights a key challenge in AI text detection: **overfitting to the generation templates of the training corpus**. Future iterations should include aggressive keyword/template sanitization and a more diverse training set.

---

## License

This project is for research and academic purposes.
