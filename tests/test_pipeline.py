import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import pytest

from ml_agent.pipeline import Pipeline
from ml_agent.evaluation.report import EvaluationReport


@pytest.fixture
def clf_data():
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(300, 6), columns=[f"f{i}" for i in range(6)])
    y = pd.Series(np.random.randint(0, 2, 300))
    return X, y


@pytest.fixture
def reg_data():
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(300, 6), columns=[f"f{i}" for i in range(6)])
    y = pd.Series(np.random.rand(300))
    return X, y


def _fast_pipeline(task: str, **kwargs) -> Pipeline:
    """테스트용 빠른 Pipeline (n_trials=2, timeout=20, top_k=2, cv=3)."""
    return Pipeline(
        task=task,
        top_k=2,
        n_trials=2,
        timeout=20,
        cv_folds=3,
        models=["random_forest", "extra_trees"],
        track=False,
        **kwargs,
    )


# ------------------------------------------------------------------
# 기본 fit / predict
# ------------------------------------------------------------------

def test_clf_fit_predict(clf_data):
    X, y = clf_data
    pipe = _fast_pipeline("classification")
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == (300,)
    assert set(np.unique(preds)).issubset({0, 1})


def test_reg_fit_predict(reg_data):
    X, y = reg_data
    pipe = _fast_pipeline("regression")
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == (300,)


def test_predict_proba(clf_data):
    X, y = clf_data
    pipe = _fast_pipeline("classification")
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (300, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_predict_proba_regression_raises(reg_data):
    X, y = reg_data
    pipe = _fast_pipeline("regression")
    pipe.fit(X, y)
    with pytest.raises(NotImplementedError):
        pipe.predict_proba(X)


# ------------------------------------------------------------------
# evaluate → EvaluationReport
# ------------------------------------------------------------------

def test_evaluate_returns_report(clf_data):
    X, y = clf_data
    pipe = _fast_pipeline("classification")
    pipe.fit(X, y)
    report = pipe.evaluate(X, y)
    assert isinstance(report, EvaluationReport)
    assert "accuracy" in report.metrics
    assert "roc_auc" in report.metrics


def test_evaluate_regression(reg_data):
    X, y = reg_data
    pipe = _fast_pipeline("regression")
    pipe.fit(X, y)
    report = pipe.evaluate(X, y)
    assert "rmse" in report.metrics
    assert "r2" in report.metrics
    assert report.metrics["rmse"] >= 0


# ------------------------------------------------------------------
# Override 인터페이스
# ------------------------------------------------------------------

def test_model_override(clf_data):
    X, y = clf_data
    pipe = Pipeline(
        task="classification",
        models=["logistic_regression"],
        top_k=1,
        n_trials=2,
        timeout=10,
        cv_folds=3,
        track=False,
    )
    pipe.fit(X, y)
    assert pipe._automl.top_k_names_ == ["logistic_regression"]


def test_ensemble_override(clf_data):
    X, y = clf_data
    pipe = Pipeline(
        task="classification",
        models=["random_forest", "extra_trees"],
        top_k=2,
        n_trials=2,
        timeout=10,
        cv_folds=3,
        ensemble="stacking",
        track=False,
    )
    pipe.fit(X, y)
    assert pipe._automl.best_ensemble_method_ in ("stacking", "single")


# ------------------------------------------------------------------
# 에러 처리
# ------------------------------------------------------------------

def test_not_fitted_raises():
    pipe = Pipeline(task="classification")
    with pytest.raises(RuntimeError, match="not fitted"):
        pipe.predict(pd.DataFrame({"a": [1, 2]}))


def test_invalid_task_raises():
    with pytest.raises(ValueError, match="task must be"):
        Pipeline(task="clustering")


def test_repr():
    pipe = Pipeline(task="classification")
    assert "not fitted" in repr(pipe)
