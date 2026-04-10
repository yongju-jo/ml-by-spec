import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import pytest

from ml_agent.automl.screener import Screener
from ml_agent.automl.tuner import Tuner
from ml_agent.automl.ensembler import Ensembler, StackingEnsemble, BlendingEnsemble
from ml_agent.automl.automl import AutoML


@pytest.fixture
def clf_data():
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(200, 6), columns=[f"f{i}" for i in range(6)])
    y = pd.Series(np.random.randint(0, 2, 200))
    return X, y


@pytest.fixture
def reg_data():
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(200, 6), columns=[f"f{i}" for i in range(6)])
    y = pd.Series(np.random.rand(200))
    return X, y


# ------------------------------------------------------------------
# Screener
# ------------------------------------------------------------------

def test_screener_returns_top_k(clf_data):
    X, y = clf_data
    screener = Screener(task="classification", top_k=2, cv_folds=3,
                        model_names=["random_forest", "extra_trees", "ridge"])
    top = screener.run(X, y)
    assert len(top) == 2
    assert all(n in ["random_forest", "extra_trees", "ridge"] for n in top)


def test_screener_scores_populated(clf_data):
    X, y = clf_data
    screener = Screener(task="classification", top_k=2, cv_folds=3,
                        model_names=["random_forest", "logistic_regression"])
    screener.run(X, y)
    assert "random_forest" in screener.scores_
    assert "logistic_regression" in screener.scores_


def test_screener_sampling(clf_data):
    """대용량 데이터 샘플링 확인."""
    X, y = clf_data
    # _SCREENING_MAX_ROWS보다 작은 데이터이므로 샘플링 없이 통과
    screener = Screener(task="classification", top_k=1, cv_folds=2,
                        model_names=["random_forest"])
    top = screener.run(X, y)
    assert len(top) == 1


def test_screener_regression(reg_data):
    X, y = reg_data
    screener = Screener(task="regression", top_k=2, cv_folds=3,
                        model_names=["random_forest", "ridge"])
    top = screener.run(X, y)
    assert len(top) == 2


# ------------------------------------------------------------------
# Tuner
# ------------------------------------------------------------------

def test_tuner_returns_fitted_models(clf_data):
    X, y = clf_data
    tuner = Tuner(task="classification", n_trials=3, timeout=30, cv_folds=3)
    results = tuner.run(["random_forest", "extra_trees"], X, y)
    assert "random_forest" in results
    assert "extra_trees" in results
    model, score = results["random_forest"]
    preds = model.predict(X)
    assert preds.shape == (200,)


def test_tuner_regression(reg_data):
    X, y = reg_data
    tuner = Tuner(task="regression", n_trials=3, timeout=30, cv_folds=3)
    results = tuner.run(["ridge"], X, y)
    model, score = results["ridge"]
    assert model.predict(X).shape == (200,)


# ------------------------------------------------------------------
# Ensembler
# ------------------------------------------------------------------

def _get_tuned_models(task, X, y, names, n_trials=3):
    tuner = Tuner(task=task, n_trials=n_trials, timeout=30, cv_folds=3)
    results = tuner.run(names, X, y)
    return [m for m, _ in results.values()]


def test_stacking_clf(clf_data):
    X, y = clf_data
    models = _get_tuned_models("classification", X, y, ["random_forest", "extra_trees"])
    s = StackingEnsemble(models, task="classification", cv_folds=3)
    s.fit(X, y)
    preds = s.predict(X)
    proba = s.predict_proba(X)
    assert preds.shape == (200,)
    assert proba.shape[1] == 2


def test_blending_clf(clf_data):
    X, y = clf_data
    models = _get_tuned_models("classification", X, y, ["random_forest", "extra_trees"])
    b = BlendingEnsemble(models, task="classification")
    b.fit(X, y)
    preds = b.predict(X)
    assert preds.shape == (200,)


def test_ensembler_auto_picks_best(clf_data):
    X, y = clf_data
    models = _get_tuned_models("classification", X, y, ["random_forest", "logistic_regression"])
    ensembler = Ensembler(task="classification", method="auto", cv_folds=3)
    ensemble = ensembler.run(models, X, y)
    assert ensembler.best_method_ in ("stacking", "blending")
    assert ensemble is not None


def test_ensembler_forced_stacking(clf_data):
    X, y = clf_data
    models = _get_tuned_models("classification", X, y, ["random_forest", "extra_trees"])
    ensembler = Ensembler(task="classification", method="stacking", cv_folds=3)
    ensemble = ensembler.run(models, X, y)
    assert ensembler.best_method_ == "stacking"


# ------------------------------------------------------------------
# AutoML (빠른 smoke test)
# ------------------------------------------------------------------

def test_automl_clf_smoke(clf_data):
    X, y = clf_data
    automl = AutoML(
        task="classification",
        top_k=2,
        n_trials=3,
        timeout=30,
        cv_folds=3,
        model_names=["random_forest", "extra_trees", "logistic_regression"],
    )
    automl.fit(X, y)
    preds = automl.predict(X)
    assert preds.shape == (200,)
    assert automl.best_model_ is not None


def test_automl_reg_smoke(reg_data):
    X, y = reg_data
    automl = AutoML(
        task="regression",
        top_k=2,
        n_trials=3,
        timeout=30,
        cv_folds=3,
        model_names=["random_forest", "ridge"],
    )
    automl.fit(X, y)
    preds = automl.predict(X)
    assert preds.shape == (200,)


def test_automl_not_fitted_raises():
    automl = AutoML(task="classification")
    with pytest.raises(RuntimeError, match="not fitted"):
        automl.predict(pd.DataFrame({"a": [1, 2]}))
