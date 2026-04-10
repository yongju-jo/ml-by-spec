"""
DataLoader: raw DataFrame을 받아 train/test 분할 및 기본 검증을 수행한다.
spec: 01_data_spec.md §2, §6
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    @property
    def train_shape(self) -> Tuple[int, int]:
        return self.X_train.shape

    @property
    def test_shape(self) -> Tuple[int, int]:
        return self.X_test.shape


class DataLoader:
    """
    raw DataFrame → train/test 분할.

    Parameters
    ----------
    test_size : float
        Test 비율. 기본 0.2 (80/20 분할).
    seed : int
        재현성을 위한 random seed. 기본 42.
    task : str
        "classification" 또는 "regression".
        classification이면 stratified split 적용.
    """

    def __init__(
        self,
        task: str,
        test_size: float = 0.2,
        seed: int = 42,
    ) -> None:
        if task not in ("classification", "regression"):
            raise ValueError(f"task must be 'classification' or 'regression', got '{task}'")
        self.task = task
        self.test_size = test_size
        self.seed = seed

    def load(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
    ) -> DataSplit:
        """
        X, y를 받아 검증 후 train/test 분할 반환.
        """
        X, y = self._validate(X, y)

        stratify = y if self.task == "classification" else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=stratify,
        )

        return DataSplit(
            X_train=X_train.reset_index(drop=True),
            X_test=X_test.reset_index(drop=True),
            y_train=y_train.reset_index(drop=True),
            y_test=y_test.reset_index(drop=True),
        )

    def _validate(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pd.DataFrame, got {type(X)}")

        if isinstance(y, np.ndarray):
            y = pd.Series(y, name="target")
        elif not isinstance(y, pd.Series):
            raise TypeError(f"y must be pd.Series or np.ndarray, got {type(y)}")

        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same length, got X={len(X)}, y={len(y)}"
            )

        if X.shape[1] == 0:
            raise ValueError("X has no feature columns")

        if y.isnull().all():
            raise ValueError("y contains only null values")

        return X, y
