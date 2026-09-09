"""AUROC/AUPRC helpers that degrade gracefully on the single-class batches/folds
that are common with rare deterioration events, instead of raising."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def safe_auprc(y_true: np.ndarray, y_score: np.ndarray):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return average_precision_score(y_true, y_score)


def per_horizon_metrics(y_true: dict, y_score: dict) -> dict:
    """y_true/y_score: {horizon_name: 1D array}. Returns {horizon_name: {auroc, auprc}}."""
    out = {}
    for horizon, true_arr in y_true.items():
        score_arr = y_score[horizon]
        out[horizon] = {
            "auroc": safe_auroc(true_arr, score_arr),
            "auprc": safe_auprc(true_arr, score_arr),
        }
    return out


def per_class_auroc(y_true: np.ndarray, y_score: np.ndarray, class_names: list) -> dict:
    """y_true/y_score: [N, n_classes]. Returns {class_name: auroc}."""
    return {
        name: safe_auroc(y_true[:, i], y_score[:, i])
        for i, name in enumerate(class_names)
    }
