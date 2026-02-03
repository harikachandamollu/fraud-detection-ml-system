from fastapi import FastAPI, HTTPException
import pandas as pd
import mlflow.pyfunc
import numpy as np
import traceback
import joblib
from pathlib import Path

from src.feature_engineering import build_features_infer

app = FastAPI()

# ---------- Config ----------
MODEL_NAME = "fraud_detector" 

BASE_DIR = Path(__file__).resolve().parent        
PROJECT_ROOT = BASE_DIR.parent                    
FE_STATE_PATH = PROJECT_ROOT / "fe_state.joblib"  

# ---------- Load artifacts ----------
try:
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/Production")
except Exception as e:
    raise RuntimeError(f"Failed to load MLflow model '{MODEL_NAME}' from Production: {repr(e)}")

try:
    fe_state = joblib.load(FE_STATE_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load feature engineering state at {FE_STATE_PATH}: {repr(e)}")


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: dict):
    try:
        # 1) Single-row JSON -> 1-row DataFrame
        raw_df = pd.DataFrame([payload])

        # 2) Apply training-fitted feature engineering (freq maps, indicators, scaler, column alignment)
        X = build_features_infer(raw_df, fe_state)

        # 3) Predict using MLflow pyfunc model
        preds = model.predict(X)

        # 4) Normalize prediction output to a single float
        if hasattr(preds, "values"):  # pandas DataFrame/Series
            preds = preds.values

        preds = np.asarray(preds)

        # If model returns probabilities with 2 columns: take positive class prob
        if preds.ndim == 2 and preds.shape[1] >= 2:
            score = float(preds[0, 1])
        else:
            score = float(preds.reshape(-1)[0])

        return {"fraud_probability": score}

    except Exception as e:
        # Print full traceback to server logs for debugging
        print("=== PREDICT ERROR ===")
        print(repr(e))
        print(traceback.format_exc())

        # Return the real exception (repr is usually more informative)
        raise HTTPException(status_code=500, detail=repr(e))


