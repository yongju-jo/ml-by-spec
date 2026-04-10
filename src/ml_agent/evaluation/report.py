"""
EvaluationReport: 지표 + 시각화를 담는 결과 객체.
spec: 05_evaluation_spec.md §5
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ml_agent.evaluation.metrics import classification_metrics, regression_metrics
from ml_agent.evaluation.plots import (
    plot_actual_vs_predicted,
    plot_confusion_matrix,
    plot_lift_chart,
    plot_model_comparison,
    plot_pr_curve,
    plot_residuals,
    plot_roc_curve,
)


class EvaluationReport:
    """
    spec: 05_evaluation_spec.md §5

    Usage
    -----
    report = pipe.evaluate(X_test, y_test)
    report.metrics             # dict
    report.plots()             # 전체 시각화 출력
    report.feature_importance()
    report.compare_models()
    report.save_html("report.html")
    """

    def __init__(
        self,
        task: str,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
        screening_scores: Dict[str, float],
        ensemble_scores: Dict[str, float],
        best_method: Optional[str],
        model=None,
        X_test: Optional[pd.DataFrame] = None,
    ) -> None:
        self.task = task
        self._y_true = y_true
        self._y_pred = y_pred
        self._y_proba = y_proba
        self.screening_scores = screening_scores
        self.ensemble_scores = ensemble_scores
        self.best_method = best_method
        self._model = model
        self._X_test = X_test

        # 지표 계산
        if task == "classification":
            self.metrics: Dict[str, float] = classification_metrics(y_true, y_pred, y_proba)
        else:
            self.metrics = regression_metrics(y_true, y_pred)

    # ------------------------------------------------------------------
    # 시각화
    # ------------------------------------------------------------------

    def plots(self, save_html: Optional[str] = None) -> None:
        """전체 시각화 출력 (콘솔 + 선택적 HTML 저장)."""
        import matplotlib.pyplot as plt

        if self.task == "classification":
            self._plot_classification(plt)
        else:
            self._plot_regression(plt)

        plt.tight_layout()
        if save_html:
            self.save_html(save_html)
        plt.show()

    def _plot_classification(self, plt) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Classification Evaluation", fontsize=14)

        plot_confusion_matrix(self._y_true, self._y_pred, ax=axes[0, 0])
        plot_model_comparison(self.screening_scores, ax=axes[0, 1])

        if self._y_proba is not None:
            is_binary = self._y_proba.ndim == 1 or self._y_proba.shape[1] == 2
            proba_1d = self._y_proba[:, 1] if (self._y_proba.ndim == 2 and is_binary) else self._y_proba
            if is_binary:
                plot_roc_curve(self._y_true, proba_1d, ax=axes[1, 0])
                plot_lift_chart(self._y_true, proba_1d, ax=axes[1, 1])
        plt.subplots_adjust(hspace=0.4)

    def _plot_regression(self, plt) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Regression Evaluation", fontsize=14)

        plot_actual_vs_predicted(self._y_true, self._y_pred, ax=axes[0])
        plot_residuals(self._y_true, self._y_pred, ax=axes[1])
        plot_model_comparison(self.screening_scores, ax=axes[2])

    def feature_importance(self, max_display: int = 20) -> None:
        """SHAP summary plot. spec: 05_evaluation_spec.md §6"""
        if self._model is None or self._X_test is None:
            print("feature_importance: model 또는 X_test가 없습니다.")
            return
        try:
            import shap
            import matplotlib.pyplot as plt

            model = self._model
            # 앙상블이면 best base model 사용
            if hasattr(model, "_fitted_bases"):
                model = model._fitted_bases[0]
            if hasattr(model, "_model"):
                model = model._model

            explainer = self._get_shap_explainer(model, self._X_test)
            shap_values = explainer(self._X_test)
            shap.summary_plot(shap_values, self._X_test, max_display=max_display, show=True)
        except ImportError:
            print("SHAP이 설치되지 않았습니다: pip install shap")
        except Exception as e:
            print(f"feature_importance 계산 실패: {e}")

    def explain(self, X_sample: pd.DataFrame) -> None:
        """단일 샘플 SHAP waterfall plot."""
        if self._model is None:
            print("explain: model이 없습니다.")
            return
        try:
            import shap
            model = self._model
            if hasattr(model, "_fitted_bases"):
                model = model._fitted_bases[0]
            if hasattr(model, "_model"):
                model = model._model

            explainer = self._get_shap_explainer(model, X_sample)
            shap_values = explainer(X_sample)
            shap.plots.waterfall(shap_values[0])
        except Exception as e:
            print(f"explain 실패: {e}")

    def compare_models(self) -> None:
        """Screening 및 앙상블 비교 출력."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        plot_model_comparison(self.screening_scores, ax=axes[0])
        if self.ensemble_scores:
            axes[1].bar(list(self.ensemble_scores.keys()), list(self.ensemble_scores.values()))
            axes[1].set_title(f"Stacking vs Blending (선택: {self.best_method})")
            axes[1].set_ylabel("CV Score")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # 저장
    # ------------------------------------------------------------------

    def save_html(self, path: str) -> None:
        """평가 결과를 HTML 파일로 저장."""
        import matplotlib.pyplot as plt
        import io, base64

        html_parts = [
            "<html><head><meta charset='utf-8'></head><body>",
            f"<h1>ML Agent Evaluation Report</h1>",
            f"<h2>Task: {self.task} | Best method: {self.best_method}</h2>",
            "<h2>Metrics</h2><table border='1' cellpadding='5'>",
            "<tr><th>Metric</th><th>Value</th></tr>",
        ]
        for k, v in self.metrics.items():
            html_parts.append(f"<tr><td>{k}</td><td>{v:.4f}</td></tr>")
        html_parts.append("</table>")

        # 플롯 임베드
        buf = io.BytesIO()
        if self.task == "classification":
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            self._plot_classification(plt)
        else:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            self._plot_regression(plt)
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=100)
        plt.close()
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode()
        html_parts.append(f"<img src='data:image/png;base64,{img_b64}'/>")
        html_parts.append("</body></html>")

        Path(path).write_text("\n".join(html_parts), encoding="utf-8")
        print(f"HTML 저장 완료: {path}")

    def __repr__(self) -> str:
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        return f"EvaluationReport(task={self.task}, {metrics_str})"

    # ------------------------------------------------------------------
    # 내부 유틸
    # ------------------------------------------------------------------

    @staticmethod
    def _get_shap_explainer(model, X: pd.DataFrame):
        import shap
        model_type = type(model).__name__.lower()
        if any(t in model_type for t in ("forest", "tree", "boost", "gbm", "xgb", "lgbm", "catboost")):
            return shap.TreeExplainer(model)
        if any(t in model_type for t in ("linear", "ridge", "lasso", "logistic")):
            return shap.LinearExplainer(model, X)
        # DL / 기타: KernelExplainer with sampling
        bg = shap.sample(X, min(100, len(X)))
        return shap.KernelExplainer(model.predict, bg)
