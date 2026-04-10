# 03. Model Spec

## 1. 지원 모델 목록

### 1.1 Sklearn 계열

| 모델 | Task |
|------|------|
| LogisticRegression | Classification |
| LinearRegression / Ridge / Lasso | Regression |
| RandomForestClassifier / Regressor | Both |
| ExtraTreesClassifier / Regressor | Both |
| SVC / SVR | Both |
| KNeighborsClassifier / Regressor | Both |

### 1.2 Boosting 계열

| 모델 | Task |
|------|------|
| XGBoostClassifier / Regressor | Both |
| LightGBMClassifier / Regressor | Both |
| CatBoostClassifier / Regressor | Both |

### 1.3 Deep Learning 계열 (PyTorch)

| 모델 | Task | 설명 |
|------|------|------|
| MLP | Both | 기본 다층 퍼셉트론 |
| TabNet | Both | Attention 기반 tabular 특화 모델 |
| FT-Transformer | Both | Feature Tokenizer + Transformer |

## 2. 모델 선택 전략 (AutoML)

```
1. 모든 모델을 기본 하이퍼파라미터로 빠르게 학습 (Screening)
2. 상위 K개 모델에 대해 Bayesian Optimization으로 튜닝
3. 튜닝된 모델로 Stacking / Blending 앙상블 구성
4. CV 기반으로 최종 모델 또는 앙상블 선택
```

### 상위 K 기준
- 기본값: K=3
- Override 가능

## 3. 하이퍼파라미터 튜닝

- **방식:** Bayesian Optimization (Optuna 사용)
- **탐색 범위:** 모델별 사전 정의된 search space
- **평가 기준:** Cross-Validation score (Classification: ROC-AUC, Regression: RMSE)
- **시간 예산:** trial 수 또는 시간 제한으로 제어 (기본 TBD)

## 4. 앙상블

### Stacking (기본)
- Base learners: 상위 K개 튜닝 모델
- Meta learner: LogisticRegression (classification) / Ridge (regression)
- OOF (Out-of-Fold) 예측으로 meta feature 생성

### Blending (비교군)
- Base learners: 상위 K개 튜닝 모델
- Holdout set으로 meta feature 생성
- Stacking과 CV 성능 비교 후 표준 결정

## 5. Override 인터페이스

```python
# 특정 모델만 사용
pipe = Pipeline(task="classification", models=["xgboost", "lightgbm"])

# 앙상블 방식 지정
pipe = Pipeline(task="classification", ensemble="blending")

# 튜닝 예산 조정
pipe = Pipeline(task="classification", n_trials=50)

# 모델 직접 주입
from sklearn.ensemble import GradientBoostingClassifier
pipe = Pipeline(task="classification", custom_models=[GradientBoostingClassifier()])
```

## 6. 확정 사항

| 항목 | 결정 |
|------|------|
| Bayesian Opt 종료 조건 | `n_trials=100` 또는 `timeout=300s` 중 먼저 도달한 조건으로 종료 |
| Deep learning 학습 | Background에서 비동기 실행, 완료 시 결과만 반환 |
| Meta learner Override | 허용 (`meta_learner` 파라미터로 주입 가능) |
