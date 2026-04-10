# specml

Spec-Driven AutoML pipeline for tabular data.

Preprocessing, model selection, Bayesian tuning, and ensemble — all automatic.
Benchmarked on [Grinsztajn et al. (2022)](https://arxiv.org/abs/2207.08815) datasets.

---

## Philosophy

This project is built with **Spec Driven Development (SDD)**.
All design decisions are written as specs in [`specs/`](specs/) before any code is written.
The implementation is a direct translation of those specs.

---

## Quickstart

```python
from ml_agent import Pipeline

pipe = Pipeline(task="classification")
pipe.fit(X_train, y_train)

report = pipe.evaluate(X_test, y_test)
print(report.metrics)

y_pred = pipe.predict(X_new)
```

---

## Features

| Stage | What it does |
|-------|-------------|
| **Auto Preprocessing** | Detects column types (numeric, categorical, binary, datetime) and applies appropriate imputation, encoding, and scaling per model family |
| **Screening** | Trains all models with default params using 5-fold CV, selects top K |
| **Tuning** | Bayesian Optimization (Optuna) on top K models — stops at `n_trials=100` or `timeout=300s` |
| **Ensemble** | Builds Stacking and Blending, compares CV scores, picks the better one |
| **Evaluation** | ROC-AUC, Lift chart, SHAP feature importance, HTML report export |

---

## Models

**Sklearn** — RandomForest, ExtraTrees, LogisticRegression, Ridge, Lasso, KNN, SVM

**Boosting** — XGBoost, LightGBM, CatBoost

**Deep Learning** — MLP (PyTorch, early stopping)

---

## Installation

```bash
git clone https://github.com/whdydwn1/ml-by-spec.git
cd ml-by-spec
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

> **macOS (Apple Silicon / Intel):** XGBoost and LightGBM require OpenMP.
> ```bash
> install_name_tool -add_rpath \
>   $(python -c "import torch; print(torch.__file__.replace('__init__.py','lib'))") \
>   .venv/lib/python3.9/site-packages/xgboost/lib/libxgboost.dylib
> ```

---

## Usage

### Basic

```python
from ml_agent import Pipeline

pipe = Pipeline(task="regression")
pipe.fit(X, y)
report = pipe.evaluate(X_test, y_test)
report.plots()
```

### Override

```python
pipe = Pipeline(
    task="classification",
    models=["xgboost", "lightgbm"],   # 특정 모델만
    top_k=2,
    ensemble="stacking",              # 앙상블 방식 고정
    n_trials=50,
    timeout=120,
)
```

### Evaluation

```python
report = pipe.evaluate(X_test, y_test)

report.metrics                  # {"roc_auc": 0.91, "accuracy": 0.85, ...}
report.plots()                  # 전체 시각화
report.feature_importance()     # SHAP summary plot
report.compare_models()         # Screening 모델 비교
report.save_html("report.html") # HTML 저장
```

---

## Benchmark

Datasets from [Grinsztajn et al. (2022) — *Why tree-based models still outperform deep learning on tabular data*](https://arxiv.org/abs/2207.08815).

```bash
python benchmarks/download_datasets.py   # 데이터셋 다운로드 (~32개)
```

---

## Project Structure

```
specml/
├── specs/               # Spec 문서 (SDD)
├── benchmarks/          # Grinsztajn 데이터셋 로더
└── src/ml_agent/
    ├── pipeline.py      # 통합 진입점
    ├── data/            # TypeDetector, DataLoader
    ├── preprocessing/   # AutoPreprocessor
    ├── models/          # sklearn / boosting / DL + search spaces
    ├── automl/          # Screener → Tuner → Ensembler
    └── evaluation/      # Metrics, Plots, EvaluationReport
```

---

## Tests

```bash
pytest tests/
```

---

## Specs

| 파일 | 내용 |
|------|------|
| [`specs/00_problem_spec.md`](specs/00_problem_spec.md) | 목표, 성공 기준, Scope |
| [`specs/01_data_spec.md`](specs/01_data_spec.md) | Input/Output 계약, Grinsztajn 데이터셋 목록 |
| [`specs/02_preprocessing_spec.md`](specs/02_preprocessing_spec.md) | 타입별 전처리 파이프라인 |
| [`specs/03_model_spec.md`](specs/03_model_spec.md) | 지원 모델, 하이퍼파라미터 튜닝 |
| [`specs/04_automl_spec.md`](specs/04_automl_spec.md) | AutoML 전체 흐름, 앙상블 전략 |
| [`specs/05_evaluation_spec.md`](specs/05_evaluation_spec.md) | 평가 지표, 시각화, XAI |
