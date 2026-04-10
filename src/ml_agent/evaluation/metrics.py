"""
평가 지표 계산 모듈.
spec: 05_evaluation_spec.md §2
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)


def classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> Dict[str, float]:
    """
    분류 평가 지표 계산.
    spec: 05_evaluation_spec.md §2 Classification
    """
    result: Dict[str, float] = {}

    result["accuracy"] = accuracy_score(y_true, y_pred)
    result["f1"] = f1_score(y_true, y_pred, average="binary" if len(np.unique(y_true)) == 2 else "macro", zero_division=0)
    result["precision"] = precision_score(y_true, y_pred, average="binary" if len(np.unique(y_true)) == 2 else "macro", zero_division=0)
    result["recall"] = recall_score(y_true, y_pred, average="binary" if len(np.unique(y_true)) == 2 else "macro", zero_division=0)

    if y_proba is not None:
        is_binary = y_proba.ndim == 1 or y_proba.shape[1] == 2
        proba = y_proba[:, 1] if (y_proba.ndim == 2 and is_binary) else y_proba
        multi_class = "raise" if is_binary else "ovr"

        try:
            result["roc_auc"] = roc_auc_score(y_true, proba, multi_class=multi_class if not is_binary else "raise")
        except Exception:
            result["roc_auc"] = float("nan")

        try:
            result["pr_auc"] = average_precision_score(y_true, proba if is_binary else proba.max(axis=1))
        except Exception:
            result["pr_auc"] = float("nan")

        try:
            result["log_loss"] = log_loss(y_true, y_proba)
        except Exception:
            result["log_loss"] = float("nan")

    return result


def regression_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    회귀 평가 지표 계산.
    spec: 05_evaluation_spec.md §2 Regression
    """
    result: Dict[str, float] = {}

    result["rmse"] = root_mean_squared_error(y_true, y_pred)
    result["mae"] = mean_absolute_error(y_true, y_pred)
    result["r2"] = r2_score(y_true, y_pred)

    try:
        result["mape"] = mean_absolute_percentage_error(y_true, y_pred)
    except Exception:
        result["mape"] = float("nan")

    return result
