import pandas as pd
from sklearn.preprocessing import StandardScaler


def drop_constant_features(df: pd.DataFrame):
    """Remove constant or near-constant features"""
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    return df.drop(columns=constant_cols), constant_cols


def frequency_encode(df: pd.DataFrame, cat_cols):
    """
    Replace categorical values with their frequency.
    Works well for high-cardinality features.
    """
    freq_maps = {}

    for col in cat_cols:
        freq = df[col].value_counts(normalize=True)
        df[col] = df[col].map(freq)
        freq_maps[col] = freq

    return df, freq_maps

def add_missing_indicators(df: pd.DataFrame, threshold=0.3):
    """
    Create binary missing-value indicators for features
    with high missing ratios (efficient version).
    """
    missing_ratio = df.isnull().mean()
    indicator_cols = missing_ratio[missing_ratio > threshold].index.tolist()

    if not indicator_cols:
        return df, []

    indicators = (
        df[indicator_cols]
        .isnull()
        .astype(int)
        .add_suffix("_missing")
    )

    df = pd.concat([df, indicators], axis=1)

    return df, indicator_cols


def build_features(df: pd.DataFrame, target: str):
    """
    Full feature engineering pipeline
    """
    df = df.copy()

    # Separate target
    y = df[target]
    X = df.drop(columns=[target])

    # Drop constant features
    X, dropped_cols = drop_constant_features(X)

    # Identify categorical columns
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    # Frequency encoding
    X, freq_maps = frequency_encode(X, cat_cols)

    # Missing indicators
    X, indicator_cols = add_missing_indicators(X)

    # Fill remaining missing values
    X = X.fillna(0)

    # Scale features
    scaler = StandardScaler()
    X[X.columns] = scaler.fit_transform(X)

    metadata = {
        "dropped_columns": dropped_cols,
        "categorical_columns": cat_cols,
        "indicator_columns": indicator_cols,
        "scaler": scaler,
        "frequency_maps": freq_maps
    }

    return X, y, metadata
