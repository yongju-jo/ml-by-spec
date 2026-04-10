"""
Boosting 계열 모델 래퍼 (XGBoost, LightGBM, CatBoost).
spec: 03_model_spec.md §1.2
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_agent.models.base import BaseModel, Task


class XGBoostModel(BaseModel):
    name = "xgboost"
    family = "boosting"

    def __init__(self, task: Task, **kwargs) -> None:
        super().__init__(task, **kwargs)
        from xgboost import XGBClassifier, XGBRegressor
        kwargs.setdefault("random_state", 42)
        kwargs.setdefault("n_jobs", -1)
        kwargs.setdefault("verbosity", 0)
        kwargs.setdefault("eval_metric", "logloss" if task == "classification" else "rmse")
        cls = XGBClassifier if task == "classification" else XGBRegressor
        self._model = cls(**kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task == "regression":
            raise NotImplementedError
        return self._model.predict_proba(X)

    def get_params(self) -> dict:
        return self._model.get_params()

    def set_params(self, **params) -> "XGBoostModel":
        self._model.set_params(**params)
        return self


class LightGBMModel(BaseModel):
    name = "lightgbm"
    family = "boosting"

    def __init__(self, task: Task, **kwargs) -> None:
        super().__init__(task, **kwargs)
        from lightgbm import LGBMClassifier, LGBMRegressor
        kwargs.setdefault("random_state", 42)
        kwargs.setdefault("n_jobs", -1)
        kwargs.setdefault("verbose", -1)
        cls = LGBMClassifier if task == "classification" else LGBMRegressor
        self._model = cls(**kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task == "regression":
            raise NotImplementedError
        return self._model.predict_proba(X)

    def get_params(self) -> dict:
        return self._model.get_params()

    def set_params(self, **params) -> "LightGBMModel":
        self._model.set_params(**params)
        return self


class CatBoostModel(BaseModel):
    name = "catboost"
    family = "catboost"

    def __init__(self, task: Task, **kwargs) -> None:
        super().__init__(task, **kwargs)
        from catboost import CatBoostClassifier, CatBoostRegressor
        kwargs.setdefault("random_seed", 42)
        kwargs.setdefault("verbose", 0)
        cls = CatBoostClassifier if task == "classification" else CatBoostRegressor
        self._model = cls(**kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CatBoostModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task == "regression":
            raise NotImplementedError
        return self._model.predict_proba(X)

    def get_params(self) -> dict:
        return self._model.get_params()

    def set_params(self, **params) -> "CatBoostModel":
        self._model.set_params(**params)
        return self
