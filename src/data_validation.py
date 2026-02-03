import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
TARGET_COL = "isFraud"
ID_COL = "TransactionID"
MAX_MISSING_THRESHOLD = 0.95
MIN_POSITIVE_RATE = 0.01

ARTIFACT_DIR = Path("../data/validation")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_TARGET_VALUES = {0, 1}

# Validation Functions
def validate_schema(df: pd.DataFrame, target: str):
    errors = []

    if target not in df.columns:
        errors.append(f"Target column '{target}' not found")

    if df.columns.duplicated().any():
        errors.append("Duplicate column names detected")

    return errors


def validate_ids(df: pd.DataFrame, id_col: str):
    errors = []

    if id_col not in df.columns:
        errors.append(f"ID column '{id_col}' missing")
        return errors

    if df[id_col].isnull().any():
        errors.append("Null values found in TransactionID")

    if df[id_col].duplicated().any():
        errors.append("Duplicate TransactionID values detected")

    return errors


def validate_target(df: pd.DataFrame, target: str):
    errors = []

    if df[target].isnull().any():
        errors.append("Target contains missing values")

    unique_values = set(df[target].unique())
    if not unique_values.issubset(EXPECTED_TARGET_VALUES):
        errors.append(f"Unexpected target values found: {unique_values}")

    return errors


def validate_missingness(df: pd.DataFrame, threshold: float):
    missing_ratio = df.isnull().mean()
    high_missing_cols = missing_ratio[missing_ratio > threshold].index.tolist()
    return high_missing_cols, missing_ratio


def validate_ranges(df: pd.DataFrame):
    warnings = {}

    if "TransactionAmt" in df.columns:
        warnings["TransactionAmt_negative"] = int(
            (df["TransactionAmt"] < 0).sum()
        )

    return warnings


def validate_class_imbalance(df: pd.DataFrame, target: str, min_rate: float):
    positive_rate = df[target].mean()

    if positive_rate < min_rate:
        return f"Extreme class imbalance detected: positive rate = {positive_rate:.4f}"

    return None


# Report Generation
def generate_report(df: pd.DataFrame, target: str):
    report = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "constant_features": [
            col for col in df.columns if df[col].nunique() <= 1
        ],
        "target_distribution": df[target].value_counts(normalize=True).to_dict()
    }
    return report


# Main (Fail-Fast)
if __name__ == "__main__":
    # Load data
    df_transactions = pd.read_csv("data/raw/train_transaction.csv")
    df_identity = pd.read_csv("data/raw/train_identity.csv")

    df = df_transactions.merge(
        df_identity, on=ID_COL, how="left"
    )

    # Run validations
    errors = []
    errors += validate_schema(df, TARGET_COL)
    errors += validate_ids(df, ID_COL)
    errors += validate_target(df, TARGET_COL)

    imbalance_warning = validate_class_imbalance(
        df, TARGET_COL, MIN_POSITIVE_RATE
    )

    high_missing_cols, missing_ratio = validate_missingness(
        df, MAX_MISSING_THRESHOLD
    )

    range_warnings = validate_ranges(df)
    report = generate_report(df, TARGET_COL)

    # Save validation artifacts
    missing_ratio.sort_values(ascending=False).to_csv(
        ARTIFACT_DIR / "missing_ratio.csv"
    )

    pd.Series(report["target_distribution"]).to_csv(
        ARTIFACT_DIR / "target_distribution.csv"
    )

    # Fail-fast behavior
    if errors:
        print("❌ Data Validation Errors:")
        for err in errors:
            print("-", err)
        raise ValueError("Data validation failed. Fix errors before proceeding.")

    # Console Summary
    print("✅ Data validation passed")
    print("Rows:", report["rows"], "Columns:", report["columns"])
    print("Target distribution:", report["target_distribution"])
    print("High missing features (>95%):", high_missing_cols[:10])
    print("Range warnings:", range_warnings)

    if imbalance_warning:
        print("⚠️", imbalance_warning)
