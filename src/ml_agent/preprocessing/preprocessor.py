"""
AutoPreprocessor: 컬럼 타입별 전처리를 모델 계열에 따라 자동 적용.
spec: 02_preprocessing_spec.md §3, §4, §5
"""

from __future__ import annotations

from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)

from ml_agent.data.type_detector import ColumnTypeMap
from ml_agent.preprocessing.config import PreprocessorConfig
from ml_agent.preprocessing.transformers import (
    DatetimeDecomposer,
    IQRCapper,
    TopKCategoryEncoder,
)

# 모델 계열 타입
ModelFamily = Literal["tree", "boosting", "catboost", "linear", "dl"]

# 고유값 수 기준: 이 이상이면 고차원 범주형으로 판단
_HIGH_CARDINALITY_THRESHOLD = 5


class AutoPreprocessor:
    """
    타입 감지 결과(ColumnTypeMap)를 받아 모델 계열에 맞는 전처리 파이프라인을 구성.

    Parameters
    ----------
    config : PreprocessorConfig
        전처리 설정. None이면 기본값 사용.
    """

    def __init__(self, config: Optional[PreprocessorConfig] = None) -> None:
        self.config = config or PreprocessorConfig()
        self._pipeline: Optional[SklearnPipeline] = None
        self._binary_encoders: dict = {}
        self._datetime_cols: List[str] = []
        self._type_map: Optional[ColumnTypeMap] = None
        self._model_family: Optional[ModelFamily] = None

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        type_map: ColumnTypeMap,
        model_family: ModelFamily,
    ) -> pd.DataFrame:
        self._type_map = type_map
        self._model_family = model_family

        X = self._apply_datetime(X, fit=True)
        X = self._apply_binary(X, fit=True)
        self._pipeline = self._build_pipeline(X, y, type_map, model_family)
        transformed = self._pipeline.fit_transform(X, y)
        return self._to_dataframe(transformed, X)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._pipeline is None:
            raise RuntimeError("AutoPreprocessor is not fitted. Call fit_transform first.")
        X = self._apply_datetime(X, fit=False)
        X = self._apply_binary(X, fit=False)
        transformed = self._pipeline.transform(X)
        return self._to_dataframe(transformed, X)

    # ------------------------------------------------------------------
    # Datetime: 파이프라인 전에 별도 처리 (컬럼 수가 바뀌어 ColumnTransformer와 분리)
    # ------------------------------------------------------------------

    def _apply_datetime(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        if fit:
            self._datetime_cols = self._type_map.datetime if self._type_map else []
            self._datetime_decomposer = DatetimeDecomposer(self._datetime_cols)
            self._datetime_decomposer.fit(X)
        if self._datetime_cols:
            X = self._datetime_decomposer.transform(X)
        return X

    # ------------------------------------------------------------------
    # Binary: LabelEncoding 별도 처리 (sklearn LabelEncoder는 2D 미지원)
    # ------------------------------------------------------------------

    def _apply_binary(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        binary_cols = self._type_map.binary if self._type_map else []
        if fit:
            self._binary_encoders = {}
            for col in binary_cols:
                le = LabelEncoder()
                non_null = X[col].dropna().astype(str)
                le.fit(non_null)
                self._binary_encoders[col] = le

        X = X.copy()
        for col, le in self._binary_encoders.items():
            filled = X[col].fillna(X[col].mode()[0]).astype(str)
            X[col] = le.transform(filled)
        return X

    # ------------------------------------------------------------------
    # 메인 파이프라인 구성
    # ------------------------------------------------------------------

    def _build_pipeline(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        type_map: ColumnTypeMap,
        model_family: ModelFamily,
    ) -> SklearnPipeline:
        # datetime/binary는 이미 처리됨 → 남은 컬럼만 대상
        handled = set(type_map.datetime) | set(type_map.binary)
        numeric_cols = [c for c in type_map.numeric if c in X.columns]
        cat_cols     = [c for c in type_map.categorical if c in X.columns and c not in handled]

        transformers = []

        # Numeric 파이프라인
        if numeric_cols:
            transformers.append(
                ("numeric", self._numeric_pipeline(model_family), numeric_cols)
            )

        # Categorical 파이프라인 (고차원 처리 포함)
        if cat_cols:
            low_card, high_card = self._split_cardinality(X, cat_cols)

            if low_card:
                transformers.append(
                    ("cat_low", self._categorical_pipeline(model_family, high_card=False, y=y), low_card)
                )
            if high_card:
                transformers.append(
                    ("cat_high", self._categorical_pipeline(model_family, high_card=True, y=y), high_card)
                )

        ct = ColumnTransformer(transformers=transformers, remainder="passthrough")
        return SklearnPipeline([("column_transformer", ct)])

    def _split_cardinality(
        self, X: pd.DataFrame, cols: List[str]
    ) -> tuple[List[str], List[str]]:
        low, high = [], []
        for col in cols:
            n_unique = X[col].nunique(dropna=True)
            if n_unique > self.config.ohe_threshold:
                high.append(col)
            else:
                low.append(col)
        return low, high

    # ------------------------------------------------------------------
    # 서브 파이프라인 팩토리
    # ------------------------------------------------------------------

    def _numeric_pipeline(self, model_family: ModelFamily) -> SklearnPipeline:
        steps = [("imputer", self._make_numeric_imputer())]

        if self.config.outlier_capping:
            steps.append(("capper", IQRCapper()))

        scaler = self._make_scaler(model_family)
        if scaler is not None:
            steps.append(("scaler", scaler))

        return SklearnPipeline(steps)

    def _categorical_pipeline(
        self,
        model_family: ModelFamily,
        high_card: bool,
        y: pd.Series,
    ) -> SklearnPipeline:
        steps = [("imputer", self._make_categorical_imputer())]

        if high_card:
            steps.append(("top_k", TopKCategoryEncoder(top_k=self.config.high_cardinality_top_k)))

        steps.append(("encoder", self._make_categorical_encoder(model_family, y)))
        return SklearnPipeline(steps)

    # ------------------------------------------------------------------
    # 개별 컴포넌트 팩토리
    # ------------------------------------------------------------------

    def _make_numeric_imputer(self) -> SimpleImputer:
        strategy = self.config.numeric_imputer
        if strategy == "constant":
            return SimpleImputer(strategy="constant", fill_value=self.config.numeric_fill_value)
        return SimpleImputer(strategy=strategy)  # "median" or "mean"

    def _make_categorical_imputer(self) -> SimpleImputer:
        if self.config.categorical_imputer == "missing":
            return SimpleImputer(strategy="constant", fill_value="missing")
        return SimpleImputer(strategy="most_frequent")

    def _make_scaler(self, model_family: ModelFamily):
        # Tree / Boosting / CatBoost: 스케일링 없음
        if model_family in ("tree", "boosting", "catboost"):
            return None

        s = self.config.scaler
        if s == "none":
            return None
        if s == "minmax":
            return MinMaxScaler()
        if s == "robust":
            return RobustScaler()
        return StandardScaler()  # default

    def _make_categorical_encoder(self, model_family: ModelFamily, y: pd.Series):
        if model_family == "catboost":
            # CatBoost 내부 처리 위임 → OrdinalEncoding으로 인덱스만 부여
            return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

        if model_family in ("tree", "boosting"):
            return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

        if model_family == "dl":
            return TargetEncoder(random_state=self.config.seed)

        # linear: OHE
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    # ------------------------------------------------------------------
    # 출력 DataFrame 복원
    # ------------------------------------------------------------------

    def _to_dataframe(self, array: np.ndarray, X_ref: pd.DataFrame) -> pd.DataFrame:
        try:
            feature_names = self._pipeline.named_steps[
                "column_transformer"
            ].get_feature_names_out()
        except Exception:
            feature_names = [f"f{i}" for i in range(array.shape[1])]

        return pd.DataFrame(array, columns=feature_names)
