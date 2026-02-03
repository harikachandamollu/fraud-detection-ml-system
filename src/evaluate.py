import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve


def evaluate_model(model, X_val, y_val):
    probs = model.predict_proba(X_val)[:, 1]

    roc_auc = roc_auc_score(y_val, probs)

    precision, recall, thresholds = precision_recall_curve(y_val, probs)

    results = {
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "thresholds": thresholds
    }

    return results