"""
컬럼 타입 자동 감지 모듈.
spec: 02_preprocessing_spec.md §2
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd


# 감지된 타입 레이블
NUMERIC = "numeric"
CATEGORICAL = "categorical"
BINARY = "binary"
DATETIME = "datetime"

_DATE_PATTERN = re.compile(
    r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}"  # YYYY-MM-DD 또는 YYYY/MM/DD
)

@dataclass
class ColumnTypeMap:
    numeric: List[str] = field(default_factory=list)
    categorical: List[str] = field(default_factory=list)
    binary: List[str] = field(default_factory=list)
    datetime: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, str]:
        result = {}
        for col in self.numeric:
            result[col] = NUMERIC
        for col in self.categorical:
            result[col] = CATEGORICAL
        for col in self.binary:
            result[col] = BINARY
        for col in self.datetime:
            result[col] = DATETIME
        return result

    def __repr__(self) -> str:
        return (
            f"ColumnTypeMap("
            f"numeric={len(self.numeric)}, "
            f"categorical={len(self.categorical)}, "
            f"binary={len(self.binary)}, "
            f"datetime={len(self.datetime)})"
        )


class TypeDetector:
    """
    DataFrame의 각 컬럼 타입을 자동 감지한다.

    Parameters
    ----------
    categorical_threshold : int
        고유값 수가 이 값 이하이면 int/float 컬럼도 categorical로 감지.
        기본값 20.
    """

    def __init__(self, categorical_threshold: int = 20) -> None:
        self.categorical_threshold = categorical_threshold

    def detect(self, X: pd.DataFrame) -> ColumnTypeMap:
        type_map = ColumnTypeMap()

        for col in X.columns:
            series = X[col]
            detected = self._detect_column(series)

            if detected == NUMERIC:
                type_map.numeric.append(col)
            elif detected == CATEGORICAL:
                type_map.categorical.append(col)
            elif detected == BINARY:
                type_map.binary.append(col)
            elif detected == DATETIME:
                type_map.datetime.append(col)

        return type_map

    def _detect_column(self, series: pd.Series) -> str:
        n_unique = series.nunique(dropna=True)

        # Binary: 고유값이 2개
        if n_unique == 2:
            return BINARY

        # Datetime: datetime dtype 또는 날짜 패턴 문자열
        if pd.api.types.is_datetime64_any_dtype(series):
            return DATETIME
        if pd.api.types.is_object_dtype(series):
            if self._looks_like_datetime(series):
                return DATETIME
            return CATEGORICAL

        # Numeric dtype이지만 고유값이 적으면 categorical
        if pd.api.types.is_numeric_dtype(series):
            if n_unique <= self.categorical_threshold:
                return CATEGORICAL
            return NUMERIC

        return CATEGORICAL

    @staticmethod
    def _looks_like_datetime(series: pd.Series) -> bool:
        sample = series.dropna().head(50).astype(str)
        if len(sample) == 0:
            return False
        matches = sample.apply(lambda v: bool(_DATE_PATTERN.match(v)))
        return matches.mean() >= 0.8
