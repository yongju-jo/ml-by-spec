"""
커스텀 sklearn 호환 Transformer들.
spec: 02_preprocessing_spec.md §3
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DatetimeDecomposer(BaseEstimator, TransformerMixin):
    """
    datetime 컬럼을 year, month, day, weekday, hour 파생 변수로 분해.
    spec: 02_preprocessing_spec.md §3.4
    """

    PARTS = ["year", "month", "day", "weekday", "hour"]

    def __init__(self, columns: List[str]) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None) -> "DatetimeDecomposer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            dt = pd.to_datetime(X[col], errors="coerce")
            X[f"{col}_year"]    = dt.dt.year
            X[f"{col}_month"]   = dt.dt.month
            X[f"{col}_day"]     = dt.dt.day
            X[f"{col}_weekday"] = dt.dt.weekday
            X[f"{col}_hour"]    = dt.dt.hour
            X.drop(columns=[col], inplace=True)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(self.PARTS)


class TopKCategoryEncoder(BaseEstimator, TransformerMixin):
    """
    고차원 범주형 컬럼의 상위 K개 카테고리를 유지하고 나머지를 'other'로 대체.
    spec: 02_preprocessing_spec.md §7 (확정 사항)
    """

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k
        self.top_categories_: dict = {}

    def fit(self, X, y=None) -> "TopKCategoryEncoder":
        X = self._to_df(X)
        for col in X.columns:
            top = (
                X[col]
                .value_counts()
                .head(self.top_k)
                .index.tolist()
            )
            self.top_categories_[col] = set(top)
        return self

    def transform(self, X) -> np.ndarray:
        X = self._to_df(X).copy()
        for col in X.columns:
            top = self.top_categories_.get(col, set())
            X[col] = X[col].where(X[col].isin(top), other="other")
        return X.to_numpy()

    @staticmethod
    def _to_df(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])


class IQRCapper(BaseEstimator, TransformerMixin):
    """
    수치형 컬럼에 IQR 기반 이상치 Capping 적용.
    spec: 02_preprocessing_spec.md §3.1
    """

    def __init__(self, factor: float = 1.5) -> None:
        self.factor = factor
        self.lower_: dict = {}
        self.upper_: dict = {}

    def fit(self, X, y=None) -> "IQRCapper":
        X = self._to_df(X)
        for col in X.columns:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            self.lower_[col] = q1 - self.factor * iqr
            self.upper_[col] = q3 + self.factor * iqr
        return self

    def transform(self, X) -> np.ndarray:
        X = self._to_df(X).copy()
        for col in X.columns:
            X[col] = X[col].clip(
                lower=self.lower_[col],
                upper=self.upper_[col],
            )
        return X.to_numpy()

    @staticmethod
    def _to_df(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])
