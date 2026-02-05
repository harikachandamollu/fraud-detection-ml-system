import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import lightgbm as lgb

from src.feature_engineering import build_features_train, build_features_infer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

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

    MLRUNS_DIR = PROJECT_ROOT / "mlruns"
    FE_STATE_PATH = PROJECT_ROOT / "fe_state.joblib"
    mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.as_posix()}")
    mlflow.set_experiment("fraud_detection")

    if mlflow.active_run() is not None:
        mlflow.end_run()

    # Load data
    df_trans = pd.read_csv("data/raw/train_transaction.csv")
    df_id = pd.read_csv("data/raw/train_identity.csv")
    df = df_trans.merge(df_id, on="TransactionID", how="left")

    # Train/validation split (on raw df) — avoids leakage
    df_train, df_val = train_test_split(
        df,
        test_size=0.2,
        stratify=df["isFraud"],
        random_state=RANDOM_STATE
    )

    # Feature engineering
    # Fit FE on TRAIN only, transform VAL using fitted state
    X_train, y_train, fe_state = build_features_train(df_train, target="isFraud")
    joblib.dump(fe_state, FE_STATE_PATH)

    # NOTE: df_val still has isFraud; build_features_infer will drop it safely,
    # but we pass without target anyway for clarity.
    X_val = build_features_infer(df_val.drop(columns=["isFraud"]), fe_state)
    y_val = df_val["isFraud"]

    # Logistic Regression (Baseline)
    with mlflow.start_run(run_name="logistic_regression"):
        mlflow.log_artifact(str(FE_STATE_PATH), artifact_path="preprocessing")
        model, metrics = train_logistic_regression(X_train, y_train, X_val, y_val)
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

    # LightGBM (Final)
    with mlflow.start_run(run_name="lightgbm"):
        mlflow.log_artifact(str(FE_STATE_PATH), artifact_path="preprocessing")
        model, metrics = train_lightgbm(X_train, y_train, X_val, y_val)
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")


if __name__ == "__main__":
    main()
