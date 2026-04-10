# 01. Data Spec

## 1. 목표

Pipeline의 Input/Output 데이터 계약을 정의하고,
벤치마크에 사용할 Grinsztajn 데이터셋 목록과 로딩 방식을 명세한다.

## 2. Pipeline Input 스펙

```python
pipe.fit(X_train, y_train)
pipe.predict(X_new)
```

| 항목 | 타입 | 조건 |
|------|------|------|
| `X_train` | `pd.DataFrame` | 컬럼명 필수, index 무관 |
| `y_train` | `pd.Series` or `np.ndarray` | 1D, X_train과 길이 동일 |
| `X_new` | `pd.DataFrame` | fit 시 컬럼명과 동일해야 함 |

- 사용자는 전처리 없이 raw DataFrame을 그대로 넣는다.
- 컬럼 타입 감지는 Pipeline 내부에서 자동 수행한다.

## 3. Pipeline Output 스펙

| 메서드 | 반환 타입 | 설명 |
|--------|-----------|------|
| `predict(X)` | `np.ndarray` (1D) | 최종 예측값 |
| `predict_proba(X)` | `np.ndarray` (2D) | 클래스 확률 (classification only) |
| `evaluate(X, y)` | `EvaluationReport` | 지표 + 시각화 객체 |

## 4. Grinsztajn 벤치마크 데이터셋

- 출처: Grinsztajn et al. (2022) "Why tree-based models still outperform deep learning on tabular data"
- 데이터 소스: OpenML (API로 자동 다운로드)
- 총 45개 데이터셋 (classification 30개 + regression 15개)

### Classification 데이터셋 (주요)

| Dataset | OpenML ID | 샘플 수 | 피처 수 |
|---------|-----------|---------|---------|
| electricity | 44120 | 38,474 | 8 |
| covertype | 44121 | 566,602 | 54 |
| pol | 44122 | 10,082 | 48 |
| house_16H | 44123 | 22,784 | 16 |
| kdd_ipums_la_97-small | 44124 | 5,188 | 20 |
| MagicTelescope | 44125 | 13,376 | 11 |
| bank-marketing | 44126 | 10,578 | 8 |
| phoneme | 44127 | 3,172 | 5 |
| MiniBooNE | 44128 | 72,998 | 50 |
| jannis | 44129 | 57,580 | 54 |

### Regression 데이터셋 (주요)

| Dataset | OpenML ID | 샘플 수 | 피처 수 |
|---------|-----------|---------|---------|
| cpu_act | 44132 | 8,192 | 21 |
| pol | 44133 | 10,082 | 48 |
| elevators | 44134 | 16,599 | 18 |
| wine_quality | 44136 | 6,497 | 11 |
| Ailerons | 44137 | 13,750 | 40 |
| houses | 44138 | 20,640 | 8 |
| house_16H | 44139 | 22,784 | 16 |
| diamonds | 44140 | 53,940 | 9 |
| Brazilian_houses | 44141 | 10,692 | 11 |
| Bike_Sharing_Demand | 44142 | 17,379 | 12 |

> 전체 목록은 `benchmarks/dataset_registry.py`에 정의

## 5. 데이터 로딩 방식

```python
from ml_agent.benchmarks import GrinsztajnLoader

loader = GrinsztajnLoader()
X, y = loader.load("electricity")          # 이름으로 로드
X, y = loader.load(openml_id=44120)       # OpenML ID로 로드
datasets = loader.load_all(task="classification")  # 전체 로드
```

- OpenML Python API (`openml` 패키지) 사용
- 최초 다운로드 후 로컬 캐시 (`~/.cache/ml_agent/datasets/`)에 저장
- 캐시 존재 시 재다운로드 없이 로컬에서 로드

## 6. 데이터 분할 기준

| 항목 | 기준 |
|------|------|
| Train / Test 분할 | 80 / 20 (기본), stratified |
| CV fold | 5-fold (Screening & Tuning) |
| Blending holdout | Train의 20% |
| Random seed | 고정 (`seed=42` 기본, Override 가능) |
