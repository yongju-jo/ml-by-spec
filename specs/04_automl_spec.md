# 04. AutoML Spec

## 1. 목표

사용자 개입 없이 모델 Screening → 튜닝 → 앙상블 → 최종 선택까지 자동으로 수행한다.

## 2. 전체 흐름

```
fit(X_train, y_train)
        │
        ▼
[1] Screening
    모든 모델을 기본 하이퍼파라미터로 빠르게 Cross-Validation
    → CV score 기준 상위 K개 선택 (기본 K=3)
        │
        ▼
[2] Tuning
    상위 K개 모델에 Bayesian Optimization 적용
    - n_trials=100 또는 timeout=300s 중 먼저 만족하는 조건으로 종료
    - DL 모델은 Background에서 비동기 실행
        │
        ▼
[3] Ensemble
    튜닝된 K개 모델로 Stacking & Blending 각각 구성
    → CV score 비교 후 더 나은 방식을 최종 앙상블로 선택
        │
        ▼
[4] 최종 선택
    앙상블 vs 단일 Best 모델 성능 비교
    → 더 나은 쪽을 pipe.predict()의 기본 모델로 등록
```

## 3. Screening 단계

- CV fold: 기본 5-fold Stratified (classification) / KFold (regression)
- 평가 지표: ROC-AUC (classification) / RMSE (regression)
- 목적: 빠른 탈락 필터링 (전체 데이터의 일부 샘플링 가능, TBD)

## 4. Tuning 단계

- 라이브러리: **Optuna**
- Sampler: TPESampler (기본)
- Pruner: MedianPruner (중간 성능 나쁜 trial 조기 종료)
- 종료 조건: `n_trials=100` OR `timeout=300s`
- 각 모델별 search space는 `models/search_spaces.py`에 정의

### DL 모델 비동기 처리
- Screening/Tuning 중 DL 모델은 별도 프로세스에서 실행
- 완료 시 결과를 메인 프로세스로 전달하여 Ensemble 단계에 합류
- DL 완료 전에 non-DL 앙상블이 먼저 구성될 수 있음 (DL 합류 후 재비교)

## 5. Ensemble 단계

### Stacking
- Base learners: 튜닝된 K개 모델
- Meta learner: LogisticRegression (classification) / Ridge (regression) — Override 가능
- OOF(Out-of-Fold) 예측으로 meta feature 생성 (leakage 방지)

### Blending
- Base learners: 튜닝된 K개 모델
- Holdout set (train의 20%): meta feature 생성에 사용
- 나머지 80%로 base learner 학습

### 비교 및 선택
- Stacking CV score vs Blending CV score 비교
- 높은 쪽을 최종 앙상블로 채택
- 결과는 실험 로그에 기록

## 6. 실험 추적

- 도구: **MLflow** (로컬 tracking server)
- 기록 항목:
  - 각 모델의 Screening score
  - 각 모델의 Tuning 결과 (best params, best score)
  - Stacking vs Blending 비교 결과
  - 최종 선택 모델 및 성능

```python
# 자동으로 MLflow experiment 생성
pipe = Pipeline(task="classification", experiment_name="my_experiment")
pipe.fit(X_train, y_train)
# → mlflow ui 로 결과 확인 가능
```

## 7. Override 인터페이스

```python
# K 조정
pipe = Pipeline(task="classification", top_k=5)

# 특정 모델만 사용
pipe = Pipeline(task="classification", models=["xgboost", "lightgbm"])

# 튜닝 예산 조정
pipe = Pipeline(task="classification", n_trials=50, timeout=120)

# 앙상블 방식 고정
pipe = Pipeline(task="classification", ensemble="stacking")

# Meta learner 변경
from sklearn.linear_model import LogisticRegression
pipe = Pipeline(task="classification", meta_learner=LogisticRegression(C=0.1))

# 실험 추적 비활성화
pipe = Pipeline(task="classification", track=False)
```

## 8. 확정 사항

| 항목 | 결정 |
|------|------|
| Screening 샘플링 | 데이터가 일정 기준 초과 시 자동 샘플링 (수행 최적 사이즈 기준 TBD, 기본 50,000건 제안) |
| DL timeout 초과 시 | 부분 학습 결과를 앙상블에 포함 (checkpoint 기준 최선 가중치 사용) |
