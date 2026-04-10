# 00. Problem Spec

## 1. 목표

tabular 데이터에 대한 **범용 AutoML End-to-End 파이프라인 패키지** 개발.
사용자는 데이터를 넣으면 전처리, 모델 선택, 앙상블, 평가까지 자동으로 수행된다.

## 2. 지원 Task

| Task | 설명 |
|------|------|
| `classification` | 이진 및 다중 클래스 분류 |
| `regression` | 연속값 예측 |

## 3. 핵심 사용자 인터페이스

```python
from ml_agent import Pipeline

pipe = Pipeline(task="classification")  # or "regression"
pipe.fit(X_train, y_train)
report = pipe.evaluate(X_test, y_test)
y_pred = pipe.predict(X_new)
```

- 사용자가 전처리, 모델, 하이퍼파라미터를 직접 지정하지 않아도 동작해야 한다.
- 필요 시 각 단계를 override 할 수 있는 인터페이스도 제공한다.

## 4. 성공 기준 (벤치마크)

- **데이터셋:** Grinsztajn et al. (2022) 벤치마크 (OpenML 기반 tabular 데이터셋)
- **비교 기준:** 논문에서 보고된 각 데이터셋별 tree-based / deep learning SOTA 성능
- **목표:** 벤치마크 데이터셋의 평균 성능이 논문의 Random Forest baseline 이상

| Task | 주요 지표 |
|------|-----------|
| Classification | ROC-AUC, Accuracy |
| Regression | RMSE, R² |

## 5. 범위 (Scope)

### In Scope
- Tabular 정형 데이터 (수치형 + 범주형 혼합)
- 자동 전처리 (결측치, 인코딩, 스케일링)
- 다중 모델 학습 및 자동 선택/앙상블
- 분류 / 회귀 지원
- Grinsztajn 벤치마크 재현 실험

### Out of Scope
- 시계열, 이미지, 텍스트 데이터
- 온라인 학습 (Incremental learning)
- 모델 서빙 / 배포 인프라

## 6. 확정 사항

| 항목 | 결정 |
|------|------|
| 하이퍼파라미터 튜닝 | Bayesian Optimization |
| 앙상블 기본값 | Stacking (Blending과 성능 비교 후 표준 확정) |
| Deep learning 프레임워크 | PyTorch |
| 실험 추적 | 도입 (도구 미정) |
| Override 인터페이스 | 제공 (전처리, 모델, 앙상블 각 단계 커스텀 가능) |
