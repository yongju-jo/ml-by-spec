"""
Sklearn 계열 모델 래퍼.
spec: 03_model_spec.md §1.1
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import (
    Lasso, LogisticRegression, Ridge, LinearRegression,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

from ml_agent.models.base import BaseModel, Task


class _SklearnWrapper(BaseModel):
    """sklearn estimator를 BaseModel 인터페이스로 래핑하는 내부 기반 클래스."""

    _clf_cls = None
    _reg_cls = None
    family = "tree"  # 서브클래스에서 override

    def __init__(self, task: Task, **kwargs) -> None:
        super().__init__(task, **kwargs)
        cls = self._clf_cls if task == "classification" else self._reg_cls
        self._model = cls(**kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_SklearnWrapper":
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self._model, "predict_proba"):
            raise NotImplementedError
        return self._model.predict_proba(X)

    def get_params(self) -> dict:
        return self._model.get_params()

    def set_params(self, **params) -> "_SklearnWrapper":
        self._model.set_params(**params)
        return self


# ------------------------------------------------------------------
# Tree 계열
# ------------------------------------------------------------------

class RandomForestModel(_SklearnWrapper):
    name = "random_forest"
    family = "tree"
    _clf_cls = RandomForestClassifier
    _reg_cls = RandomForestRegressor

    def __init__(self, task: Task, **kwargs) -> None:
        kwargs.setdefault("n_jobs", -1)
        kwargs.setdefault("random_state", 42)
        super().__init__(task, **kwargs)


class ExtraTreesModel(_SklearnWrapper):
    name = "extra_trees"
    family = "tree"
    _clf_cls = ExtraTreesClassifier
    _reg_cls = ExtraTreesRegressor

    def __init__(self, task: Task, **kwargs) -> None:
        kwargs.setdefault("n_jobs", -1)
        kwargs.setdefault("random_state", 42)
        super().__init__(task, **kwargs)


# ------------------------------------------------------------------
# Linear 계열
# ------------------------------------------------------------------

class LogisticRegressionModel(_SklearnWrapper):
    name = "logistic_regression"
    family = "linear"
    _clf_cls = LogisticRegression
    _reg_cls = Ridge  # regression fallback

    def __init__(self, task: Task, **kwargs) -> None:
        kwargs.setdefault("max_iter", 1000)
        kwargs.setdefault("random_state", 42)
        super().__init__(task, **kwargs)


class RidgeModel(_SklearnWrapper):
    name = "ridge"
    family = "linear"
    _clf_cls = LogisticRegression
    _reg_cls = Ridge


class LassoModel(_SklearnWrapper):
    name = "lasso"
    family = "linear"
    _clf_cls = LogisticRegression  # classification fallback
    _reg_cls = Lasso


# ------------------------------------------------------------------
# Instance-based
# ------------------------------------------------------------------

class KNNModel(_SklearnWrapper):
    name = "knn"
    family = "linear"  # 스케일링 필요
    _clf_cls = KNeighborsClassifier
    _reg_cls = KNeighborsRegressor

    def __init__(self, task: Task, **kwargs) -> None:
        kwargs.setdefault("n_jobs", -1)
        super().__init__(task, **kwargs)


class SVMModel(_SklearnWrapper):
    name = "svm"
    family = "linear"
    _clf_cls = SVC
    _reg_cls = SVR

    def __init__(self, task: Task, **kwargs) -> None:
        kwargs.setdefault("probability", True)  # predict_proba 활성화 (SVC)
        super().__init__(task, **kwargs)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task == "regression":
            raise NotImplementedError
        return self._model.predict_proba(X)
