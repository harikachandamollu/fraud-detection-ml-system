# Fraud Detection ML System

A structured machine learning project for learning and practicing **end-to-end ML workflows** using a fraud detection use case.
This repository focuses on building good habits around data handling, model training, and experiment tracking.

---

## 🎯 Project Goal

The goal of this project is to:

* Practice working with an **imbalanced classification problem**
* Learn how to structure ML code beyond notebooks
* Understand model evaluation and experiment tracking using MLflow
* Gain hands-on experience with common steps in the ML lifecycle

This project is primarily **learning-oriented** and designed to reflect how ML projects are organized in real teams.

---

## 🔄 Pipeline Overview

Raw Data
   ↓
Data Validation
   ↓
EDA & Feature Engineering
   ↓
Model Training (Baseline → Final)
   ↓
Evaluation & Threshold Selection
   ↓
Drift Detection
   ↓
Model Registry (MLflow)
   ↓
Inference API (FastAPI)

---

## 📊 Dataset

* IEEE-CIS Fraud Detection dataset
* Highly imabalanced (~3% fraud)
* High-dimensional tabular data
* Significant missing values

## ⚠️ Notes on Data

* The dataset is used as a realistic proxy for financial transaction data. 
* Dataset files are excluded from version control
* Paths assume data exists locally under `data/raw/`

---

## 🔍 Exploratory Data Analysis (EDA)

EDA is performed to:

* Understand target imbalance
* Identify missing-value patterns
* Distinguish numerical vs categorical features
* Highlight potential data quality issues

EDA is kept separate from production code and documented in:
* `notebooks/eda.ipynb`
* `docs/eda_summary.md`

---

## 🧠 Key Choices (Learning-focused)

* **Target:** Fraud vs Non-Fraud (binary classification)
* **Primary metric:** ROC-AUC
* **Secondary metric:** Recall at a fixed threshold
* **Validation strategy:** Stratified train/validation split
* **Baseline model:** Logistic Regression
* **Tree-based model:** LightGBM
* **Experiment tracking:** MLflow (SQLite backend)

Design decisions and reasoning are documented in `DECISIONS.md`.

--- 

## 🧱 Feature Engineering

Feature engineering focuses on **simple, robust techniques** suitable for tabular data:

* Dropping constant / near-constant features
* Frequency encoding for categorical variables
* Missing-value indicator features
* Filling remaining missing values
* Feature scaling where appropriate

The feature engineering pipeline is implemented in `src/feature_engineering.py`.

---

## 🧪 Model Training

Two models are trained and compared:

* **Baseline:** Logistic Regression  
* **Final model:** LightGBM

Training includes:
* Stratified train/validation split
* Handling class imbalance
* Logging parameters and metrics to MLflow

Training logic is implemented in `src/train.py`.

---

## 📈 Model Evaluation

Because of class imbalance:
* Accuracy is not used
* ROC-AUC is the primary metric
* Precision-recall trade-offs are analyzed

Instead of relying on a default probability cutoff, threshold selection is treated as a **deliberate decision**.

Evaluation and threshold logic are implemented in:
* `src/evaluate.py`
* `src/threshold_analysis.py`

---

## 📉 Data Drift Detection

To explore how models can degrade over time, basic **data drift detection** is implemented using:

* Kolmogorov–Smirnov (KS) test
* Population Stability Index (PSI)

These techniques help identify distribution changes in incoming data and are implemented in `src/drift_detection.py`.

This is intended as a learning exercise, not a full monitoring system.

---

## 📊 Experiment Tracking (MLflow)

All experiments are tracked using **MLflow**, including:

* Model parameters
* Evaluation metrics
* Trained model artifacts

The MLflow UI can be locally to inspect runs and compare models.

Start the MLflow UI:

```bash
mlflow ui
```

Then open in your browser:

```
http://127.0.0.1:5000
```

---

## 🌐 Inference API (FastAPI)

A simple inference service is implemented to demonstrate how a trained model can be used in an application:

* Built using FastAPI
* Loads the selected model from MLflow
* Exposes a `/predict` endpoint
* Includes automatic Swagger documentation

API code lives in `src/api/main.py`

This steps is intended to bridge model training and application integration.

---

## 📁 Project Structure

```
fraud-detection-ml-system/
│
├── data/
│   └── raw/                     # Raw data (not tracked in Git)
│
├── src/
│   ├── api/
│   │   └── main.py              # FastAPI inference service
│   ├── data_validation.py       # Basic data quality checks
│   ├── feature_engineering.py   # Feature pipeline
│   ├── train.py                 # Model training + MLflow logging
│   ├── evaluate.py              # Evaluation utilities
│   ├── threshold_analysis.py    # Threshold selection logic
│   ├── drift_detection.py       # Data drift checks
│   └── load_model.py            # MLflow model loading helper
│
├── notebooks/
│   └── eda.ipynb
│
├── docs/
│   ├── eda_notes.md
│   ├── feature_engineering_plan.md
│   ├── training_decision.md
│   ├── evaluation_decisions.md
│   ├── model_promotion.md
│   ├── model_selection.md
│   └── deployment_decisions.md
│
├── DECISIONS.md                 # High-level design decisions
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Setup

### 1️⃣ Create environment

```bash
conda create -n fraud-ml python=3.10 -y
conda activate fraud-ml
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Training Pipeline

```bash
python src/train.py
```

This will:

* Load and merge transaction and identity data
* Build features
* Train a baseline Logistic Regression model
* Train a LightGBM model
* Log metrics, parameters, and models to MLflow

---

## 🔮 Possible Next Steps

Planned or optional extensions include:

* Hyperparameter tuning
* Automated retraining based on drift
* Model explainability (e.g. SHAP)
* Containerization and cloud deployment
These are intentionally left out of the current scope.

---

## 👤 Author

**Harika Reddy Chandamollu**
Learning-focused ML / Data Science projects

---

This repository is part of an ongoing learning journey into applied machine learning.