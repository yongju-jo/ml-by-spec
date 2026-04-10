"""
평가 시각화 모듈.
spec: 05_evaluation_spec.md §3
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _ensure_matplotlib():
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        return plt, sns
    except ImportError:
        raise ImportError("matplotlib and seaborn are required for plotting.")


def plot_roc_curve(y_true, y_proba, ax=None):
    from sklearn.metrics import roc_curve, auc
    plt, sns = _ensure_matplotlib()

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    return ax


def plot_pr_curve(y_true, y_proba, ax=None):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    plt, _ = _ensure_matplotlib()

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, label=f"PR AUC = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    return ax


def plot_confusion_matrix(y_true, y_pred, ax=None):
    from sklearn.metrics import confusion_matrix
    plt, sns = _ensure_matplotlib()

    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    return ax


def plot_lift_chart(y_true, y_proba, n_bins: int = 10, ax=None):
    plt, _ = _ensure_matplotlib()

    df = pd.DataFrame({"y": y_true, "proba": y_proba})
    df["decile"] = pd.qcut(df["proba"], n_bins, labels=False, duplicates="drop")
    df["decile"] = df["decile"].max() - df["decile"]  # 높은 확률이 1번 decile

    baseline = df["y"].mean()
    lift = df.groupby("decile")["y"].mean() / baseline

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.bar(lift.index + 1, lift.values)
    ax.axhline(1.0, color="red", linestyle="--", label="Baseline")
    ax.set_xlabel("Decile")
    ax.set_ylabel("Lift")
    ax.set_title("Lift Chart")
    ax.legend()
    return ax


def plot_actual_vs_predicted(y_true, y_pred, ax=None):
    plt, _ = _ensure_matplotlib()

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.4, s=10)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    return ax


def plot_residuals(y_true, y_pred, ax=None):
    plt, sns = _ensure_matplotlib()

    residuals = np.array(y_true) - np.array(y_pred)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(residuals, kde=True, ax=ax)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Residual")
    ax.set_title("Residual Distribution")
    return ax


def plot_model_comparison(scores: Dict[str, float], ax=None):
    """Screening 단계 모델별 score 비교 bar chart."""
    plt, _ = _ensure_matplotlib()

    names = list(scores.keys())
    vals = list(scores.values())
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, len(names)), 4))
    ax.barh(names, vals)
    ax.set_xlabel("CV Score")
    ax.set_title("Model Comparison (Screening)")
    return ax
