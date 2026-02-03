import pandas as pd
from sklearn.preprocessing import StandardScaler


def drop_constant_features(df: pd.DataFrame):
    """Remove constant or near-constant features"""
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    return df.drop(columns=constant_cols), constant_cols


def frequency_encode_fit(df: pd.DataFrame, cat_cols):
    """
    Fit frequency maps for categorical columns.
    """
    freq_maps = {}
    for col in cat_cols:
        freq_maps[col] = df[col].value_counts(normalize=True, dropna=False)
    return freq_maps


def frequency_encode_transform(df: pd.DataFrame, cat_cols, freq_maps):
    """
    Apply previously-fitted frequency maps.
    Unknown categories -> 0.
    """
    df = df.copy()
    for col in cat_cols:
        # map known values; unseen -> NaN -> fill 0 later
        df[col] = df[col].map(freq_maps.get(col, {}))
    return df


def add_missing_indicators_fit(df: pd.DataFrame, threshold=0.3):
    """
    Decide which columns get missing indicators based on training data.
    """
    missing_ratio = df.isnull().mean()
    indicator_cols = missing_ratio[missing_ratio > threshold].index.tolist()
    return indicator_cols


def add_missing_indicators_transform(df: pd.DataFrame, indicator_cols):
    """
    Apply missing indicators using the selected columns from training.
    """
    df = df.copy()
    if not indicator_cols:
        return df

    indicators = (
        df[indicator_cols]
        .isnull()
        .astype(int)
        .add_suffix("_missing")
    )
    return pd.concat([df, indicators], axis=1)


def _prepare_X_base(df: pd.DataFrame, target: str | None):
    """Internal helper: separate X and y safely."""
    df = df.copy()
    y = None
    if target and target in df.columns:
        y = df[target]
        df = df.drop(columns=[target])
    return df, y


def build_features_train(df: pd.DataFrame, target: str = "isFraud", missing_threshold: float = 0.3):
    """
    TRAINING: fit everything (dropped cols, freq maps, indicator cols, scaler)
    and return X, y, state.
    """
    X, y = _prepare_X_base(df, target)

    if y is None:
        raise ValueError(f"Target column '{target}' not found in training dataframe.")

    # 1) Drop constant features (fit)
    X, dropped_cols = drop_constant_features(X)

    # 2) Identify categorical columns (fit)
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    # 3) Fit + apply frequency encoding
    freq_maps = frequency_encode_fit(X, cat_cols)
    X = frequency_encode_transform(X, cat_cols, freq_maps)

    # 4) Fit + apply missing indicators
    indicator_cols = add_missing_indicators_fit(X, threshold=missing_threshold)
    X = add_missing_indicators_transform(X, indicator_cols)

    # 5) Fill missing
    X = X.fillna(0)

    # 6) Fit scaler and transform
    scaler = StandardScaler()
    X[X.columns] = scaler.fit_transform(X)

    state = {
        "target": target,
        "dropped_columns": dropped_cols,
        "categorical_columns": cat_cols,
        "indicator_columns": indicator_cols,
        "missing_threshold": missing_threshold,
        "frequency_maps": freq_maps,
        "scaler": scaler,
        "feature_columns": list(X.columns),  # critical for stable API
    }

    return X, y, state


def build_features_infer(df: pd.DataFrame, state: dict):
    """
    INFERENCE/API: use the fitted state from training to produce X with identical columns.
    """
    X, _ = _prepare_X_base(df, state.get("target"))

    # Ensure all training-time columns exist (raw input may be missing some)
    # We do NOT add engineered cols here; those are added later (missing indicators)
    # For raw missing columns, create them as NaN so the pipeline works.
    for col in state.get("categorical_columns", []):
        if col not in X.columns:
            X[col] = pd.NA

    # Drop columns that were dropped in training (ignore if not present)
    dropped_cols = state.get("dropped_columns", [])
    X = X.drop(columns=[c for c in dropped_cols if c in X.columns], errors="ignore")

    # Apply frequency encoding using training maps
    cat_cols = state.get("categorical_columns", [])
    freq_maps = state.get("frequency_maps", {})
    X = frequency_encode_transform(X, cat_cols, freq_maps)

    # Missing indicators based on training selection
    indicator_cols = state.get("indicator_columns", [])
    # Some indicator base columns might be missing in API input; create them
    for base_col in indicator_cols:
        if base_col not in X.columns:
            X[base_col] = pd.NA
    X = add_missing_indicators_transform(X, indicator_cols)

    # Fill remaining missing values (including unseen categories -> NaN)
    X = X.fillna(0)

    # Align feature columns exactly to training
    feature_columns = state.get("feature_columns")
    if feature_columns is not None:
        X = X.reindex(columns=feature_columns, fill_value=0)

    # Scale using trained scaler
    scaler = state["scaler"]
    X[X.columns] = scaler.transform(X)

    return X
