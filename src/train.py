import pandas as pd
import mlflow
import os
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import lightgbm as lgb

from feature_engineering import build_features
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

DATA_DIR = PROJECT_ROOT / "data" / "raw"

RANDOM_STATE = 42


def train_logistic_regression(X_train, y_train, X_val, y_val):
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    preds = model.predict_proba(X_val)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y_val, preds),
        "precision": precision_score(y_val, preds > 0.5),
        "recall": recall_score(y_val, preds > 0.5),
    }

    return model, metrics


def train_lightgbm(X_train, y_train, X_val, y_val):
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)
    preds = model.predict_proba(X_val)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y_val, preds),
        "precision": precision_score(y_val, preds > 0.5),
        "recall": recall_score(y_val, preds > 0.5),
    }

    return model, metrics


def main():
    # Load data
    df_trans = pd.read_csv(DATA_DIR / "train_transaction.csv")
    df_id = pd.read_csv(DATA_DIR / "train_identity.csv")
    df = df_trans.merge(df_id, on="TransactionID", how="left")

    # Feature engineering
    X, y, _ = build_features(df, target="isFraud")

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    mlflow.set_experiment("fraud_detection")

    # Logistic Regression (Baseline)
    with mlflow.start_run(run_name="logistic_regression"):
        model, metrics = train_logistic_regression(X_train, y_train, X_val, y_val)
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

    # LightGBM (Final)
    with mlflow.start_run(run_name="lightgbm"):
        model, metrics = train_lightgbm(X_train, y_train, X_val, y_val)
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")


if __name__ == "__main__":
    main()
