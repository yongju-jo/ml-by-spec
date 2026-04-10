"""
Ensembler: Stacking / Blending 앙상블 구성 및 비교.
spec: 04_automl_spec.md §5
"""

from __future__ import annotations

import logging
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from ml_agent.models.base import BaseModel, Task

logger = logging.getLogger(__name__)

EnsembleMethod = Literal["stacking", "blending", "auto"]


class StackingEnsemble:
    """
    OOF(Out-of-Fold) 예측으로 meta feature 생성 후 meta learner 학습.
    spec: 04_automl_spec.md §5 Stacking
    """

    def __init__(
        self,
        base_models: List[BaseModel],
        task: Task,
        meta_learner=None,
        cv_folds: int = 5,
        seed: int = 42,
    ) -> None:
        self.base_models = base_models
        self.task = task
        self.cv_folds = cv_folds
        self.seed = seed
        self.meta_learner = meta_learner or self._default_meta_learner()
        self._fitted_bases: List[BaseModel] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StackingEnsemble":
        meta_X = self._build_oof_features(X, y)
        self.meta_learner.fit(meta_X, y)
        # base model 전체 데이터로 재학습
        self._fitted_bases = []
        for model in self.base_models:
            model.fit(X, y)
            self._fitted_bases.append(model)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        meta_X = self._build_test_features(X)
        return self.meta_learner.predict(meta_X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task == "regression":
            raise NotImplementedError
        meta_X = self._build_test_features(X)
        return self.meta_learner.predict_proba(meta_X)

    def cv_score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """앙상블 전체의 CV score 계산 (간소화: meta feature → meta learner CV)."""
        meta_X = self._build_oof_features(X, y)
        metric = "roc_auc" if self.task == "classification" else "neg_root_mean_squared_error"
        cv = self._make_cv(y)
        scores = cross_val_score(self.meta_learner, meta_X, y, cv=cv, scoring=metric)
        return float(scores.mean())

    def _build_oof_features(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        n = len(X)
        n_models = len(self.base_models)
        oof = np.zeros((n, n_models))
        cv = self._make_cv(y)

        for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr = y.iloc[tr_idx]

            for m_idx, model in enumerate(self.base_models):
                model.fit(X_tr, y_tr)
                if self.task == "classification":
                    try:
                        proba = model.predict_proba(X_val)
                        oof[val_idx, m_idx] = proba[:, 1] if proba.shape[1] == 2 else proba.max(axis=1)
                    except NotImplementedError:
                        oof[val_idx, m_idx] = model.predict(X_val).astype(float)
                else:
                    oof[val_idx, m_idx] = model.predict(X_val)

        return oof

    def _build_test_features(self, X: pd.DataFrame) -> np.ndarray:
        n_models = len(self._fitted_bases)
        test_meta = np.zeros((len(X), n_models))
        for m_idx, model in enumerate(self._fitted_bases):
            if self.task == "classification":
                try:
                    proba = model.predict_proba(X)
                    test_meta[:, m_idx] = proba[:, 1] if proba.shape[1] == 2 else proba.max(axis=1)
                except NotImplementedError:
                    test_meta[:, m_idx] = model.predict(X).astype(float)
            else:
                test_meta[:, m_idx] = model.predict(X)
        return test_meta

    def _default_meta_learner(self):
        if self.task == "classification":
            return LogisticRegression(max_iter=1000, random_state=self.seed)
        return Ridge(random_state=self.seed)

    def _make_cv(self, y):
        if self.task == "classification":
            return StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
        return KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)


class BlendingEnsemble:
    """
    Holdout set (train 20%)으로 meta feature 생성.
    spec: 04_automl_spec.md §5 Blending
    """

    def __init__(
        self,
        base_models: List[BaseModel],
        task: Task,
        meta_learner=None,
        holdout_ratio: float = 0.2,
        seed: int = 42,
    ) -> None:
        self.base_models = base_models
        self.task = task
        self.holdout_ratio = holdout_ratio
        self.seed = seed
        self.meta_learner = meta_learner or self._default_meta_learner()
        self._fitted_bases: List[BaseModel] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BlendingEnsemble":
        from sklearn.model_selection import train_test_split
        stratify = y if self.task == "classification" else None
        X_tr, X_hold, y_tr, y_hold = train_test_split(
            X, y, test_size=self.holdout_ratio, random_state=self.seed, stratify=stratify
        )
        # base model을 train 80%로 학습
        self._fitted_bases = []
        for model in self.base_models:
            model.fit(X_tr, y_tr)
            self._fitted_bases.append(model)

        # holdout으로 meta feature 생성
        meta_X = self._build_features(X_hold)
        self.meta_learner.fit(meta_X, y_hold)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        meta_X = self._build_features(X)
        return self.meta_learner.predict(meta_X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task == "regression":
            raise NotImplementedError
        meta_X = self._build_features(X)
        return self.meta_learner.predict_proba(meta_X)

    def cv_score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        블렌딩 평가: train/holdout 분할 후 base 학습 → holdout 예측 → meta learner 학습 → holdout score.
        state를 변경하지 않고 score만 반환한다.
        """
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, root_mean_squared_error

        stratify = y if self.task == "classification" else None
        X_tr, X_hold, y_tr, y_hold = train_test_split(
            X, y, test_size=self.holdout_ratio, random_state=self.seed, stratify=stratify
        )

        # base 모델 임시 학습 (state 변경 없이 별도 인스턴스 불필요 — 여기선 직접 사용)
        temp_bases = []
        for model in self.base_models:
            model.fit(X_tr, y_tr)
            temp_bases.append(model)

        # holdout meta feature 생성
        meta_hold = np.zeros((len(X_hold), len(temp_bases)))
        for m_idx, model in enumerate(temp_bases):
            if self.task == "classification":
                try:
                    proba = model.predict_proba(X_hold)
                    meta_hold[:, m_idx] = proba[:, 1] if proba.shape[1] == 2 else proba.max(axis=1)
                except NotImplementedError:
                    meta_hold[:, m_idx] = model.predict(X_hold).astype(float)
            else:
                meta_hold[:, m_idx] = model.predict(X_hold)

        if meta_hold.shape[1] == 0:
            return -np.inf

        # meta learner fit → holdout score
        from sklearn.linear_model import LogisticRegression, Ridge
        meta = LogisticRegression(max_iter=1000, random_state=self.seed) \
            if self.task == "classification" else Ridge(random_state=self.seed)
        meta.fit(meta_hold, y_hold)

        if self.task == "classification":
            proba = meta.predict_proba(meta_hold)[:, 1] if hasattr(meta, "predict_proba") else meta.predict(meta_hold)
            return float(roc_auc_score(y_hold, proba))
        else:
            pred = meta.predict(meta_hold)
            return float(-root_mean_squared_error(y_hold, pred))

    def _build_features(self, X: pd.DataFrame) -> np.ndarray:
        meta = np.zeros((len(X), len(self._fitted_bases)))
        for m_idx, model in enumerate(self._fitted_bases):
            if self.task == "classification":
                try:
                    proba = model.predict_proba(X)
                    meta[:, m_idx] = proba[:, 1] if proba.shape[1] == 2 else proba.max(axis=1)
                except NotImplementedError:
                    meta[:, m_idx] = model.predict(X).astype(float)
            else:
                meta[:, m_idx] = model.predict(X)
        return meta

    def _default_meta_learner(self):
        if self.task == "classification":
            return LogisticRegression(max_iter=1000, random_state=self.seed)
        return Ridge(random_state=self.seed)


class Ensembler:
    """
    Stacking과 Blending을 각각 구성하고 CV score를 비교해 최적 앙상블을 선택.
    spec: 04_automl_spec.md §5 비교 및 선택
    """

    def __init__(
        self,
        task: Task,
        method: EnsembleMethod = "auto",
        meta_learner=None,
        cv_folds: int = 5,
        seed: int = 42,
    ) -> None:
        self.task = task
        self.method = method
        self.meta_learner = meta_learner
        self.cv_folds = cv_folds
        self.seed = seed
        self.best_ensemble_ = None
        self.best_method_: Optional[str] = None
        self.scores_: Dict[str, float] = {}

    def run(
        self, base_models: List[BaseModel], X: pd.DataFrame, y: pd.Series
    ) -> "StackingEnsemble | BlendingEnsemble":
        """
        Stacking vs Blending 비교 후 더 나은 앙상블 반환.
        method="stacking" 또는 "blending"이면 해당 방식만 사용.
        """
        if self.method == "stacking":
            ensemble = self._build_stacking(base_models, X, y)
            self.best_ensemble_ = ensemble
            self.best_method_ = "stacking"
            return ensemble

        if self.method == "blending":
            ensemble = self._build_blending(base_models, X, y)
            self.best_ensemble_ = ensemble
            self.best_method_ = "blending"
            return ensemble

        # auto: 둘 다 평가
        stacking = StackingEnsemble(base_models, self.task, self.meta_learner, self.cv_folds, self.seed)
        blending = BlendingEnsemble(base_models, self.task, self.meta_learner, seed=self.seed)

        stacking_score = stacking.cv_score(X, y)
        blending_score = blending.cv_score(X, y)

        self.scores_ = {"stacking": stacking_score, "blending": blending_score}
        logger.info(f"[Ensembler] stacking={stacking_score:.4f}, blending={blending_score:.4f}")

        if stacking_score >= blending_score:
            stacking.fit(X, y)
            self.best_ensemble_ = stacking
            self.best_method_ = "stacking"
            logger.info("[Ensembler] Stacking 선택")
        else:
            blending.fit(X, y)
            self.best_ensemble_ = blending
            self.best_method_ = "blending"
            logger.info("[Ensembler] Blending 선택")

        return self.best_ensemble_

    def _build_stacking(self, base_models, X, y):
        s = StackingEnsemble(base_models, self.task, self.meta_learner, self.cv_folds, self.seed)
        s.fit(X, y)
        return s

    def _build_blending(self, base_models, X, y):
        b = BlendingEnsemble(base_models, self.task, self.meta_learner, seed=self.seed)
        b.fit(X, y)
        return b
