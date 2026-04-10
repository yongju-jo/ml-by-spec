"""
Optuna search space 정의.
각 모델의 suggest_* 호출을 반환하는 함수 모음.
spec: 03_model_spec.md §3, 04_automl_spec.md §4
"""

from __future__ import annotations

from typing import Callable, Dict

import optuna


def _rf_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators":      trial.suggest_int("n_estimators", 50, 500),
        "max_depth":         trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }


def _extra_trees_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators":      trial.suggest_int("n_estimators", 50, 500),
        "max_depth":         trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }


def _logistic_regression_space(trial: optuna.Trial) -> dict:
    return {
        "C":       trial.suggest_float("C", 1e-4, 1e2, log=True),
        "solver":  trial.suggest_categorical("solver", ["lbfgs", "saga"]),
        "penalty": trial.suggest_categorical("penalty", ["l2", None]),
    }


def _ridge_space(trial: optuna.Trial) -> dict:
    return {
        "alpha": trial.suggest_float("alpha", 1e-4, 1e3, log=True),
    }


def _lasso_space(trial: optuna.Trial) -> dict:
    return {
        "alpha": trial.suggest_float("alpha", 1e-4, 1e2, log=True),
    }


def _knn_space(trial: optuna.Trial) -> dict:
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 3, 30),
        "weights":     trial.suggest_categorical("weights", ["uniform", "distance"]),
        "p":           trial.suggest_int("p", 1, 2),
    }


def _svm_space(trial: optuna.Trial) -> dict:
    return {
        "C":      trial.suggest_float("C", 1e-3, 1e2, log=True),
        "kernel": trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"]),
        "gamma":  trial.suggest_categorical("gamma", ["scale", "auto"]),
    }


def _xgboost_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators":        trial.suggest_int("n_estimators", 50, 1000),
        "max_depth":           trial.suggest_int("max_depth", 3, 12),
        "learning_rate":       trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample":           trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":    trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight":    trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha":           trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda":          trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
    }


def _lightgbm_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators":    trial.suggest_int("n_estimators", 50, 1000),
        "max_depth":       trial.suggest_int("max_depth", 3, 12),
        "learning_rate":   trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "num_leaves":      trial.suggest_int("num_leaves", 20, 300),
        "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":       trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda":      trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "min_child_samples":trial.suggest_int("min_child_samples", 5, 100),
    }


def _catboost_space(trial: optuna.Trial) -> dict:
    return {
        "iterations":       trial.suggest_int("iterations", 50, 1000),
        "depth":            trial.suggest_int("depth", 3, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "l2_leaf_reg":      trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count":     trial.suggest_int("border_count", 32, 255),
    }


def _mlp_space(trial: optuna.Trial) -> dict:
    n_layers = trial.suggest_int("n_layers", 1, 4)
    hidden_dims = [
        trial.suggest_int(f"hidden_dim_{i}", 64, 512)
        for i in range(n_layers)
    ]
    return {
        "hidden_dims": hidden_dims,
        "dropout":     trial.suggest_float("dropout", 0.0, 0.5),
        "lr":          trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "batch_size":  trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
    }


# 모델 이름 → search space 함수 매핑
SEARCH_SPACES: Dict[str, Callable[[optuna.Trial], dict]] = {
    "random_forest":        _rf_space,
    "extra_trees":          _extra_trees_space,
    "logistic_regression":  _logistic_regression_space,
    "ridge":                _ridge_space,
    "lasso":                _lasso_space,
    "knn":                  _knn_space,
    "svm":                  _svm_space,
    "xgboost":              _xgboost_space,
    "lightgbm":             _lightgbm_space,
    "catboost":             _catboost_space,
    "mlp":                  _mlp_space,
}


def get_search_space(model_name: str) -> Callable[[optuna.Trial], dict]:
    if model_name not in SEARCH_SPACES:
        raise KeyError(f"No search space defined for model: '{model_name}'")
    return SEARCH_SPACES[model_name]
