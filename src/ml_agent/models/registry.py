"""
모델 레지스트리: 이름 → (ModelClass, family) 매핑.
AutoML이 모델을 동적으로 인스턴스화할 때 사용.
spec: 03_model_spec.md §1
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type

from ml_agent.models.base import BaseModel, ModelFamily, Task
from ml_agent.models.sklearn_models import (
    ExtraTreesModel,
    KNNModel,
    LassoModel,
    LogisticRegressionModel,
    RandomForestModel,
    RidgeModel,
    SVMModel,
)
from ml_agent.models.boosting_models import CatBoostModel, LightGBMModel, XGBoostModel


def _lazy_mlp():
    from ml_agent.models.deep_learning import MLPModel
    return MLPModel


_REGISTRY: Dict[str, Type[BaseModel]] = {
    # Sklearn - Tree
    "random_forest":        RandomForestModel,
    "extra_trees":          ExtraTreesModel,
    # Sklearn - Linear
    "logistic_regression":  LogisticRegressionModel,
    "ridge":                RidgeModel,
    "lasso":                LassoModel,
    "knn":                  KNNModel,
    "svm":                  SVMModel,
    # Boosting
    "xgboost":              XGBoostModel,
    "lightgbm":             LightGBMModel,
    "catboost":             CatBoostModel,
    # Deep Learning (lazy import — torch 선택적 의존성)
    "mlp":                  None,  # get_model_class()에서 동적 로드
}

# task별 기본 모델 목록 (Screening 시 사용)
_DEFAULT_MODELS: Dict[str, List[str]] = {
    "classification": [
        "random_forest", "extra_trees",
        "logistic_regression", "knn",
        "xgboost", "lightgbm", "catboost",
        "mlp",
    ],
    "regression": [
        "random_forest", "extra_trees",
        "ridge", "lasso", "knn",
        "xgboost", "lightgbm", "catboost",
        "mlp",
    ],
}


def get_model_class(name: str) -> Type[BaseModel]:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown model: '{name}'. Available: {list(_REGISTRY.keys())}"
        )
    # DL 모델은 lazy import
    if name == "mlp":
        return _lazy_mlp()
    return _REGISTRY[name]


def build_model(name: str, task: Task, **kwargs) -> BaseModel:
    cls = get_model_class(name)
    return cls(task=task, **kwargs)


def default_model_names(task: Task) -> List[str]:
    return list(_DEFAULT_MODELS[task])


def available_models() -> List[str]:
    return list(_REGISTRY.keys())
