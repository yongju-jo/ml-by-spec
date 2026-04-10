"""
Pipeline: 전체 ML 워크플로우의 통합 진입점.
spec: 00_problem_spec.md §3
"""

from __future__ import annotations

import logging
from typing import List, Literal, Optional

import numpy as np
import pandas as pd

from ml_agent.data.loader import DataLoader
from ml_agent.data.type_detector import TypeDetector
from ml_agent.preprocessing.config import PreprocessorConfig
from ml_agent.preprocessing.preprocessor import AutoPreprocessor
from ml_agent.automl.automl import AutoML
from ml_agent.automl.ensembler import EnsembleMethod
from ml_agent.evaluation.report import EvaluationReport

logger = logging.getLogger(__name__)

Task = Literal["classification", "regression"]


class Pipeline:
    """
    범용 AutoML End-to-End 파이프라인.

    Parameters
    ----------
    task : str
        "classification" 또는 "regression".
    top_k : int
        Screening 후 Tuning할 상위 모델 수. 기본 3.
    n_trials : int
        Optuna trial 수 상한. 기본 100.
    timeout : int
        Optuna timeout (초). 기본 300.
    cv_folds : int
        Cross-validation fold 수. 기본 5.
    ensemble : str
        "auto" | "stacking" | "blending". 기본 "auto".
    meta_learner : optional
        앙상블 meta learner Override.
    models : list[str] | None
        사용할 모델 이름 목록. None이면 전체 기본 모델 사용.
    preprocessor_config : PreprocessorConfig | None
        전처리 설정 Override.
    test_size : float
        Train/test 분할 비율. 기본 0.2.
    seed : int
        재현성 seed. 기본 42.
    experiment_name : str
        MLflow experiment 이름. 기본 "ml_agent".
    track : bool
        MLflow 실험 추적 여부. 기본 True.

    Example
    -------
    >>> pipe = Pipeline(task="classification")
    >>> pipe.fit(X_train, y_train)
    >>> report = pipe.evaluate(X_test, y_test)
    >>> y_pred = pipe.predict(X_new)
    """

    def __init__(
        self,
        task: Task,
        top_k: int = 3,
        n_trials: int = 100,
        timeout: int = 300,
        cv_folds: int = 5,
        ensemble: EnsembleMethod = "auto",
        meta_learner=None,
        models: Optional[List[str]] = None,
        preprocessor_config: Optional[PreprocessorConfig] = None,
        test_size: float = 0.2,
        seed: int = 42,
        experiment_name: str = "ml_agent",
        track: bool = True,
    ) -> None:
        if task not in ("classification", "regression"):
            raise ValueError(f"task must be 'classification' or 'regression', got '{task}'")

        self.task = task
        self.top_k = top_k
        self.n_trials = n_trials
        self.timeout = timeout
        self.cv_folds = cv_folds
        self.ensemble = ensemble
        self.meta_learner = meta_learner
        self.models = models
        self.preprocessor_config = preprocessor_config
        self.test_size = test_size
        self.seed = seed
        self.experiment_name = experiment_name
        self.track = track

        # 내부 컴포넌트 (fit 시 초기화)
        self._type_map = None
        self._preprocessors: dict = {}   # {model_family: AutoPreprocessor}
        self._automl: Optional[AutoML] = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "Pipeline":
        """
        전처리 → AutoML (Screening → Tuning → Ensemble) 전체 실행.
        """
        loader = DataLoader(task=self.task, test_size=self.test_size, seed=self.seed)
        split = loader.load(X, y)

        X_train, y_train = split.X_train, split.y_train
        logger.info(f"[Pipeline] fit 시작 | task={self.task} | train={X_train.shape}")

        # 타입 감지
        self._type_map = TypeDetector().detect(X_train)
        logger.info(f"[Pipeline] 타입 감지: {self._type_map}")

        # AutoML 실행 (내부에서 모델별 전처리 분기)
        self._automl = AutoML(
            task=self.task,
            top_k=self.top_k,
            n_trials=self.n_trials,
            timeout=self.timeout,
            cv_folds=self.cv_folds,
            ensemble=self.ensemble,
            meta_learner=self.meta_learner,
            model_names=self.models,
            seed=self.seed,
        )

        # 전처리 후 AutoML fit
        X_proc, y_proc = self._preprocess_fit(X_train, y_train)
        self._automl.fit(X_proc, y_proc)
        self._is_fitted = True

        if self.track:
            self._log_mlflow(X_proc, y_proc)

        logger.info(f"[Pipeline] fit 완료. best_method={self._automl.best_ensemble_method_}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        X_proc = self._preprocess_transform(X)
        return self._automl.predict(X_proc)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task == "regression":
            raise NotImplementedError("predict_proba is not available for regression.")
        self._check_fitted()
        X_proc = self._preprocess_transform(X)
        return self._automl.predict_proba(X_proc)

    def evaluate(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> EvaluationReport:
        """
        예측 후 EvaluationReport 반환.
        spec: 05_evaluation_spec.md §5
        """
        self._check_fitted()
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        y_pred = self.predict(X)
        y_proba = None
        if self.task == "classification":
            try:
                y_proba = self.predict_proba(X)
            except Exception:
                pass

        return EvaluationReport(
            task=self.task,
            y_true=y,
            y_pred=y_pred,
            y_proba=y_proba,
            screening_scores=self._automl.screening_scores_,
            ensemble_scores=self._automl.ensemble_scores_,
            best_method=self._automl.best_ensemble_method_,
            model=self._automl.best_model_,
            X_test=X,
        )

    # ------------------------------------------------------------------
    # 내부 전처리
    # ------------------------------------------------------------------

    def _preprocess_fit(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """대표 family(boosting)로 전처리 후 반환. 모델별 분기는 추후 확장."""
        family = "boosting"
        prep = AutoPreprocessor(config=self.preprocessor_config)
        X_proc = prep.fit_transform(X, y, self._type_map, model_family=family)
        self._preprocessors[family] = prep
        return X_proc, y

    def _preprocess_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        family = "boosting"
        prep = self._preprocessors.get(family)
        if prep is None:
            raise RuntimeError("Preprocessor not fitted.")
        return prep.transform(X)

    # ------------------------------------------------------------------
    # MLflow 추적
    # ------------------------------------------------------------------

    def _log_mlflow(self, X: pd.DataFrame, y: pd.Series) -> None:
        try:
            import mlflow
            mlflow.set_experiment(self.experiment_name)
            with mlflow.start_run():
                mlflow.log_param("task", self.task)
                mlflow.log_param("top_k", self.top_k)
                mlflow.log_param("best_method", self._automl.best_ensemble_method_)
                mlflow.log_param("n_trials", self.n_trials)
                for name, score in self._automl.screening_scores_.items():
                    mlflow.log_metric(f"screening_{name}", score)
                for name, score in self._automl.ensemble_scores_.items():
                    mlflow.log_metric(f"ensemble_{name}", score)
        except Exception as e:
            logger.warning(f"[Pipeline] MLflow 로깅 실패 (무시): {e}")

    # ------------------------------------------------------------------
    # 유틸
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Pipeline is not fitted. Call fit() first.")

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return f"Pipeline(task={self.task}, status={status})"
