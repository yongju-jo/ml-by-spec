"""
Screener: 모든 모델을 기본 파라미터로 빠르게 CV 평가하여 상위 K개 선택.
spec: 04_automl_spec.md §3
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score

from ml_agent.models.base import BaseModel, Task
from ml_agent.models.registry import build_model, default_model_names

logger = logging.getLogger(__name__)

# 대용량 데이터 Screening 샘플링 기준
_SCREENING_MAX_ROWS = 50_000

ScreeningResult = Dict[str, float]  # {model_name: cv_score}


class Screener:
    """
    Parameters
    ----------
    task : Task
    top_k : int
        상위 몇 개 모델을 선발할지. 기본 3.
    cv_folds : int
        Cross-validation fold 수. 기본 5.
    seed : int
    model_names : list[str] | None
        None이면 task 기본 모델 목록 전체 사용.
    """

    def __init__(
        self,
        task: Task,
        top_k: int = 3,
        cv_folds: int = 5,
        seed: int = 42,
        model_names: Optional[List[str]] = None,
    ) -> None:
        self.task = task
        self.top_k = top_k
        self.cv_folds = cv_folds
        self.seed = seed
        self.model_names = model_names or default_model_names(task)
        self.scores_: ScreeningResult = {}

    def run(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Screening 실행.

        Returns
        -------
        List[str]
            상위 top_k 모델 이름 목록 (score 내림차순).
        """
        X_s, y_s = self._maybe_sample(X, y)
        logger.info(f"[Screener] {len(self.model_names)}개 모델 평가 시작 (n={len(X_s)})")

        non_dl = [n for n in self.model_names if n != "mlp"]
        dl_names = [n for n in self.model_names if n == "mlp"]

        # non-DL 모델: 동기 평가
        for name in non_dl:
            score = self._cv_score(name, X_s, y_s)
            self.scores_[name] = score
            logger.info(f"  {name}: {score:.4f}")

        # DL 모델: 비동기 평가 (결과 기다리지 않고 진행, 완료 시 합류)
        dl_futures: Dict[Future, str] = {}
        if dl_names:
            executor = ThreadPoolExecutor(max_workers=len(dl_names))
            for name in dl_names:
                fut = executor.submit(self._cv_score, name, X_s, y_s)
                dl_futures[fut] = name

        # DL 결과 수집 (이미 완료된 것만)
        if dl_futures:
            for fut in as_completed(dl_futures, timeout=None):
                name = dl_futures[fut]
                try:
                    score = fut.result()
                    self.scores_[name] = score
                    logger.info(f"  {name} (DL): {score:.4f}")
                except Exception as e:
                    logger.warning(f"  {name} (DL) 실패: {e}")
                    self.scores_[name] = -np.inf

        top_k = sorted(self.scores_, key=self.scores_.get, reverse=True)[: self.top_k]
        logger.info(f"[Screener] 상위 {self.top_k}개: {top_k}")
        return top_k

    def _cv_score(self, model_name: str, X: pd.DataFrame, y: pd.Series) -> float:
        try:
            model = build_model(model_name, task=self.task)
            cv = self._make_cv(y)
            metric = "roc_auc" if self.task == "classification" else "neg_root_mean_squared_error"

            scores = cross_val_score(
                model._model,
                X,
                y,
                cv=cv,
                scoring=metric,
                n_jobs=1,
                error_score="raise",
            )
            return float(scores.mean())
        except Exception as e:
            logger.warning(f"  {model_name} CV 실패: {e}")
            return -np.inf

    def _make_cv(self, y: pd.Series):
        if self.task == "classification":
            return StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
        return KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)

    def _maybe_sample(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        if len(X) <= _SCREENING_MAX_ROWS:
            return X, y
        logger.info(f"[Screener] 대용량 데이터 샘플링: {len(X)} → {_SCREENING_MAX_ROWS}")
        idx = np.random.RandomState(self.seed).choice(len(X), _SCREENING_MAX_ROWS, replace=False)
        return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
