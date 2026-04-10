"""
BaseModel: 모든 모델 래퍼의 공통 인터페이스.
AutoML이 모델 계열에 상관없이 동일한 방식으로 모델을 다룰 수 있도록 한다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Optional

import numpy as np
import pandas as pd

ModelFamily = Literal["tree", "boosting", "catboost", "linear", "dl"]
Task = Literal["classification", "regression"]


class BaseModel(ABC):
    """
    모든 모델 래퍼의 기반 클래스.

    Attributes
    ----------
    name : str
        모델 식별자 (registry key로 사용).
    family : ModelFamily
        전처리 분기와 Screening 정렬에 사용.
    task : Task
        fit 시점에 주입됨.
    """

    name: str
    family: ModelFamily

    def __init__(self, task: Task, **kwargs) -> None:
        self.task = task
        self._model = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support predict_proba"
        )

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        from sklearn.metrics import roc_auc_score, mean_squared_error

        if self.task == "classification":
            try:
                proba = self.predict_proba(X)
                # 이진 분류: 양성 클래스 확률, 다중 분류: 그대로
                if proba.ndim == 2 and proba.shape[1] == 2:
                    proba = proba[:, 1]
                return roc_auc_score(y, proba, multi_class="ovr")
            except NotImplementedError:
                from sklearn.metrics import accuracy_score
                return accuracy_score(y, self.predict(X))
        else:
            pred = self.predict(X)
            try:
                from sklearn.metrics import root_mean_squared_error
                rmse = root_mean_squared_error(y, pred)
            except ImportError:
                rmse = mean_squared_error(y, pred, squared=False)
            return -rmse  # 높을수록 좋게 음수 반환

    def get_params(self) -> dict:
        return {}

    def set_params(self, **params) -> "BaseModel":
        if self._model is not None:
            self._model.set_params(**params)
        return self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(task={self.task})"
