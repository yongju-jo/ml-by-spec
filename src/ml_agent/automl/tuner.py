"""
Tuner: 상위 K개 모델에 Bayesian Optimization (Optuna) 적용.
spec: 04_automl_spec.md §4
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score

from ml_agent.models.base import BaseModel, Task
from ml_agent.models.registry import build_model
from ml_agent.models.search_spaces import get_search_space

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)

# spec: 04_automl_spec.md §6 확정 사항
_DEFAULT_N_TRIALS = 100
_DEFAULT_TIMEOUT = 300  # seconds

TuningResult = Dict[str, Tuple[BaseModel, float]]  # {name: (tuned_model, best_score)}


class Tuner:
    """
    Parameters
    ----------
    task : Task
    n_trials : int
        Optuna trial 수 상한. 기본 100.
    timeout : int
        초 단위 시간 상한. 기본 300s.
    cv_folds : int
    seed : int
    """

    def __init__(
        self,
        task: Task,
        n_trials: int = _DEFAULT_N_TRIALS,
        timeout: int = _DEFAULT_TIMEOUT,
        cv_folds: int = 5,
        seed: int = 42,
    ) -> None:
        self.task = task
        self.n_trials = n_trials
        self.timeout = timeout
        self.cv_folds = cv_folds
        self.seed = seed
        self.results_: TuningResult = {}

    def run(
        self, model_names: List[str], X: pd.DataFrame, y: pd.Series
    ) -> TuningResult:
        """
        model_names 목록의 각 모델을 Bayesian Optimization으로 튜닝.

        Returns
        -------
        TuningResult
            {model_name: (best_fitted_model, best_score)}
        """
        logger.info(f"[Tuner] {model_names} 튜닝 시작")

        non_dl = [n for n in model_names if n != "mlp"]
        dl_names = [n for n in model_names if n == "mlp"]

        # non-DL: 동기 튜닝
        for name in non_dl:
            model, score = self._tune_one(name, X, y)
            self.results_[name] = (model, score)
            logger.info(f"  {name}: best_score={score:.4f}")

        # DL: 비동기 튜닝 (spec: 04_automl_spec.md §4)
        if dl_names:
            executor = ThreadPoolExecutor(max_workers=len(dl_names))
            dl_futures: Dict[Future, str] = {
                executor.submit(self._tune_one, name, X, y): name
                for name in dl_names
            }
            for fut in as_completed(dl_futures, timeout=None):
                name = dl_futures[fut]
                try:
                    model, score = fut.result()
                    self.results_[name] = (model, score)
                    logger.info(f"  {name} (DL): best_score={score:.4f}")
                except Exception as e:
                    logger.warning(f"  {name} (DL) 튜닝 실패: {e}")

        return self.results_

    def _tune_one(self, name: str, X: pd.DataFrame, y: pd.Series) -> Tuple[BaseModel, float]:
        search_space_fn = get_search_space(name)
        cv = self._make_cv(y)
        metric = "roc_auc" if self.task == "classification" else "neg_root_mean_squared_error"

        def objective(trial: optuna.Trial) -> float:
            params = search_space_fn(trial)
            try:
                model = build_model(name, task=self.task, **params)
                scores = cross_val_score(
                    model._model, X, y, cv=cv, scoring=metric, n_jobs=1, error_score="raise"
                )
                return float(scores.mean())
            except Exception:
                return -np.inf

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            pruner=optuna.pruners.MedianPruner(),
        )
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout, show_progress_bar=False)

        best_params = study.best_params
        # hidden_dims 복원 (MLP 특수 처리)
        if name == "mlp" and any(k.startswith("hidden_dim_") for k in best_params):
            n_layers = best_params.pop("n_layers", None)
            hidden_dims = [best_params.pop(f"hidden_dim_{i}") for i in range(int(n_layers or 0))]
            best_params["hidden_dims"] = hidden_dims

        best_model = build_model(name, task=self.task, **best_params)
        best_model.fit(X, y)

        return best_model, study.best_value

    def _make_cv(self, y: pd.Series):
        if self.task == "classification":
            return StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
        return KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
