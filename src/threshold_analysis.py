import numpy as np

def choose_threshold(precision, recall, thresholds, min_recall=0.8):
    for p, r, t in zip(precision, recall, thresholds):
        if r >= min_recall:
            return t
    return 0.5