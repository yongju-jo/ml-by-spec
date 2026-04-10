import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import pytest

from ml_agent.models.registry import build_model, default_model_names, available_models
from ml_agent.models.search_spaces import get_search_space, SEARCH_SPACES
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


@pytest.fixture
def clf_data():
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, 100))
    return X, y


@pytest.fixture
def reg_data():
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(np.random.rand(100))
    return X, y


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

def test_available_models():
    models = available_models()
    assert "random_forest" in models
    assert "xgboost" in models
    assert "lightgbm" in models
    assert "catboost" in models
    assert "mlp" in models


def test_default_clf_models():
    names = default_model_names("classification")
    assert len(names) > 0
    assert "xgboost" in names


def test_default_reg_models():
    names = default_model_names("regression")
    assert "ridge" in names


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        build_model("unknown_model", task="classification")


# ------------------------------------------------------------------
# Sklearn 모델
# ------------------------------------------------------------------

@pytest.mark.parametrize("name", ["random_forest", "extra_trees", "logistic_regression", "knn"])
def test_sklearn_clf_fit_predict(name, clf_data):
    X, y = clf_data
    model = build_model(name, task="classification")
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (100,)


@pytest.mark.parametrize("name", ["random_forest", "extra_trees", "ridge", "lasso", "knn"])
def test_sklearn_reg_fit_predict(name, reg_data):
    X, y = reg_data
    model = build_model(name, task="regression")
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (100,)


def test_random_forest_predict_proba(clf_data):
    X, y = clf_data
    model = build_model("random_forest", task="classification")
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (100, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


# ------------------------------------------------------------------
# Boosting 모델
# ------------------------------------------------------------------

@pytest.mark.parametrize("name", ["xgboost", "lightgbm", "catboost"])
def test_boosting_clf(name, clf_data):
    X, y = clf_data
    model = build_model(name, task="classification")
    model.fit(X, y)
    preds = model.predict(X)
    proba = model.predict_proba(X)
    assert preds.shape == (100,)
    assert proba.shape[0] == 100


@pytest.mark.parametrize("name", ["xgboost", "lightgbm", "catboost"])
def test_boosting_reg(name, reg_data):
    X, y = reg_data
    model = build_model(name, task="regression")
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (100,)


# ------------------------------------------------------------------
# MLP
# ------------------------------------------------------------------

def test_mlp_clf(clf_data):
    X, y = clf_data
    model = build_model("mlp", task="classification", max_epochs=5)
    model.fit(X, y)
    preds = model.predict(X)
    proba = model.predict_proba(X)
    assert preds.shape == (100,)
    assert proba.shape == (100, 2)


def test_mlp_reg(reg_data):
    X, y = reg_data
    model = build_model("mlp", task="regression", max_epochs=5)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (100,)


# ------------------------------------------------------------------
# score()
# ------------------------------------------------------------------

def test_score_classification(clf_data):
    X, y = clf_data
    model = build_model("random_forest", task="classification")
    model.fit(X, y)
    score = model.score(X, y)
    assert 0.0 <= score <= 1.0


def test_score_regression(reg_data):
    X, y = reg_data
    model = build_model("ridge", task="regression")
    model.fit(X, y)
    score = model.score(X, y)
    assert score <= 0.0  # RMSE 음수 반환


# ------------------------------------------------------------------
# Search spaces
# ------------------------------------------------------------------

def test_all_search_spaces_valid():
    for name in SEARCH_SPACES:
        sampler = optuna.samplers.RandomSampler(seed=42)
        study = optuna.create_study(sampler=sampler)

        def objective(trial):
            params = get_search_space(name)(trial)
            assert isinstance(params, dict)
            return 0.0

        study.optimize(objective, n_trials=3)


def test_unknown_search_space_raises():
    with pytest.raises(KeyError):
        get_search_space("unknown_model")
