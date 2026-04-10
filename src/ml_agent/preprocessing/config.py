"""
PreprocessorConfig: 전처리 설정 override용 dataclass.
spec: 02_preprocessing_spec.md §6
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class PreprocessorConfig:
    # Numeric
    numeric_imputer: Literal["median", "mean", "constant"] = "median"
    numeric_fill_value: float = 0.0          # constant imputer 사용 시
    outlier_capping: bool = False            # IQR capping 기본 Off
    scaler: Literal["standard", "minmax", "robust", "none"] = "standard"

    # Categorical
    categorical_imputer: Literal["most_frequent", "missing"] = "most_frequent"
    ohe_threshold: int = 10                  # 고유값 수 기준: OHE vs OrdinalEncoding
    high_cardinality_top_k: int = 5         # 고차원 범주형: 상위 K개 유지

    # Global
    seed: int = 42
