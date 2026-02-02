import pandas as pd
from feature_engineering import build_features
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Load raw data
df_trans = pd.read_csv(DATA_DIR / "train_transaction.csv")
df_id = pd.read_csv(DATA_DIR / "train_identity.csv")

# Merge transaction + identity tables
df = df_trans.merge(df_id, on="TransactionID", how="left")

# Run feature engineering
X, y, meta = build_features(df, target="isFraud")


# BASIC SANITY CHECKS

# Shape checks
assert X.shape[0] == df.shape[0], "Row count mismatch between X and input df"
assert len(X.columns) > 0, "No features generated"

# No missing values
assert X.isnull().sum().sum() == 0, "NaNs found in feature matrix"

# Target unchanged
assert y.equals(df["isFraud"]), "Target variable was modified"

# Missing indicators exist
assert any(col.endswith("_missing") for col in X.columns), "No missing-value indicator features found"

# No constant features
constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
assert len(constant_cols) == 0, f"Constant features found: {constant_cols}"


# METADATA CHECKS
assert "dropped_columns" in meta, "Metadata missing 'dropped_columns'"
assert "categorical_columns" in meta, "Metadata missing 'categorical_columns'"
assert "indicator_columns" in meta, "Metadata missing 'indicator_columns'"
assert "scaler" in meta, "Metadata missing 'scaler'"
assert "frequency_maps" in meta, "Metadata missing 'frequency_maps'"


# PRINT SUMMARY
print("✅ Feature engineering test passed successfully")
print("X shape:", X.shape)
print("Dropped features:", len(meta["dropped_columns"]))
print("Target distribution:")
print(y.value_counts(normalize=True))

