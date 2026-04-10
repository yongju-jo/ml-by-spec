"""
AutoML: Screening → Tuning → Ensemble 전체 흐름 오케스트레이터.
spec: 04_automl_spec.md §2
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from ml_agent.models.base import BaseModel, Task
from ml_agent.models.registry import build_model
from ml_agent.automl.screener import Screener
from ml_agent.automl.tuner import Tuner
from ml_agent.automl.ensembler import Ensembler, EnsembleMethod, StackingEnsemble, BlendingEnsemble

logger = logging.getLogger(__name__)


class AutoML:
    """
    Parameters
    ----------
    task : Task
    top_k : int
        Screening 후 Tuning할 상위 모델 수. 기본 3.
    n_trials : int
        Optuna trial 수 상한. 기본 100.
    timeout : int
        Optuna timeout (초). 기본 300.
    cv_folds : int
    ensemble : EnsembleMethod
        "auto" | "stacking" | "blending". 기본 "auto".
    meta_learner : optional
        앙상블 meta learner Override.
    model_names : list[str] | None
        None이면 task 기본 모델 목록 사용.
    seed : int
    """

    def __init__(
        self,
        task: Task,
        top_k: int = 3,
        n_trials: int = 100,
        timeout: int = 300,
        cv_folds: int = 5,
        ensemble: EnsembleMethod = "auto",
        meta_learner=None,
        model_names: Optional[List[str]] = None,
        seed: int = 42,
    ) -> None:
        self.task = task
        self.top_k = top_k
        self.n_trials = n_trials
        self.timeout = timeout
        self.cv_folds = cv_folds
        self.ensemble_method = ensemble
        self.meta_learner = meta_learner
        self.model_names = model_names
        self.seed = seed

        # 결과 저장
        self.screening_scores_: dict = {}
        self.top_k_names_: List[str] = []
        self.tuning_results_: dict = {}
        self.ensemble_scores_: dict = {}
        self.best_ensemble_method_: Optional[str] = None
        self.best_model_: Optional[Union[StackingEnsemble, BlendingEnsemble, BaseModel]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AutoML":
        """
        [1] Screening → [2] Tuning → [3] Ensemble → [4] 최종 선택
        """
        # [1] Screening
        logger.info("=" * 50)
        logger.info("[AutoML] Step 1: Screening")
        screener = Screener(
            task=self.task,
            top_k=self.top_k,
            cv_folds=self.cv_folds,
            seed=self.seed,
            model_names=self.model_names,
        )
        self.top_k_names_ = screener.run(X, y)
        self.screening_scores_ = screener.scores_

        # [2] Tuning
        logger.info("=" * 50)
        logger.info(f"[AutoML] Step 2: Tuning {self.top_k_names_}")
        tuner = Tuner(
            task=self.task,
            n_trials=self.n_trials,
            timeout=self.timeout,
            cv_folds=self.cv_folds,
            seed=self.seed,
        )
        self.tuning_results_ = tuner.run(self.top_k_names_, X, y)
        tuned_models = [model for model, _ in self.tuning_results_.values()]

        # [3] Ensemble
        logger.info("=" * 50)
        logger.info("[AutoML] Step 3: Ensemble")
        ensembler = Ensembler(
            task=self.task,
            method=self.ensemble_method,
            meta_learner=self.meta_learner,
            cv_folds=self.cv_folds,
            seed=self.seed,
        )
        ensemble = ensembler.run(tuned_models, X, y)
        self.ensemble_scores_ = ensembler.scores_
        self.best_ensemble_method_ = ensembler.best_method_

        # [4] 앙상블 vs 단일 Best 비교
        logger.info("=" * 50)
        logger.info("[AutoML] Step 4: 최종 선택")
        self.best_model_ = self._select_best(ensemble, tuned_models, X, y)

        logger.info(f"[AutoML] 완료. best_method={self.best_ensemble_method_}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.best_model_ is None:
            raise RuntimeError("AutoML is not fitted. Call fit() first.")
        return self.best_model_.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task == "regression":
            raise NotImplementedError
        if self.best_model_ is None:
            raise RuntimeError("AutoML is not fitted. Call fit() first.")
        return self.best_model_.predict_proba(X)

    def _select_best(
        self,
        ensemble,
        tuned_models: List[BaseModel],
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Union[StackingEnsemble, BlendingEnsemble, BaseModel]:
        """앙상블 CV score vs 단일 최고 모델 비교."""
        metric = "roc_auc" if self.task == "classification" else "neg_root_mean_squared_error"
        from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score

        cv = (
            StratifiedKFold(self.cv_folds, shuffle=True, random_state=self.seed)
            if self.task == "classification"
            else KFold(self.cv_folds, shuffle=True, random_state=self.seed)
        )

        best_single_score = max(score for _, score in self.tuning_results_.values())
        best_single_model = max(self.tuning_results_.values(), key=lambda x: x[1])[0]

        ensemble_score = ensemble.cv_score(X, y)

        logger.info(f"  단일 최고 모델 score: {best_single_score:.4f}")
        logger.info(f"  앙상블 score:         {ensemble_score:.4f}")

        if ensemble_score >= best_single_score:
            logger.info("  → 앙상블 선택")
            return ensemble
        else:
            logger.info("  → 단일 모델 선택")
            self.best_ensemble_method_ = "single"
            return best_single_model
