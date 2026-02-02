# Fraud Detection ML System

A structured machine learning project for learning and practicing **end-to-end ML workflows** using a fraud detection use case.
This repository focuses on building good habits around data handling, model training, and experiment tracking.

---

## 🎯 Project Goal

The goal of this project is to:

* Practice working with an **imbalanced classification problem**
* Learn how to structure ML code beyond notebooks
* Understand model evaluation and experiment tracking using MLflow

This project is primarily **learning-oriented** and designed to reflect how ML projects are organized in real teams.

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

## 📁 Project Structure

```
fraud-detection-ml-system/
│
├── data/
│   └── raw/               # Raw data (not tracked in Git)
│
├── src/
│   ├── data_validation.py # Basic data quality checks
│   ├── feature_engineering.py
│   ├── train.py           # Model training + MLflow logging
│   └── __init__.py
│
├── DECISIONS.md           # Notes on modeling & design decisions
├── requirements.txt      # Project dependencies
├── .gitignore
├── mlflow.db              # Local MLflow database (not tracked)
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

## 📊 Experiment Tracking (MLflow)

Start the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open in your browser:

```
http://127.0.0.1:5000
```

You can inspect:

* Experiments
* Model metrics
* Logged parameters

---

## ⚠️ Notes on Data

* Dataset files are excluded from version control
* Paths assume data exists locally under `data/raw/`
* This setup reflects common industry constraints around data sharing

---

## 🔮 Possible Next Steps

* Feature scaling for Logistic Regression
* Hyperparameter tuning
* Model comparison improvements
* Simple API for inference

---

## 👤 Author

**Harika Reddy Chandamollu**
Learning-focused ML / Data Science projects

---

This repository is part of an ongoing learning journey into applied machine learning.
