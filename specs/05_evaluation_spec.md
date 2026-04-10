# 05. Evaluation Spec

## 1. 목표

모델 성능을 비즈니스 관점과 통계 관점 모두에서 평가하고,
현업이 이해할 수 있는 시각화와 함께 리포트를 자동 출력한다.

## 2. 평가 지표

### Classification

| 지표 | 설명 |
|------|------|
| ROC-AUC | 기본 선택 지표 (모델 비교 기준) |
| PR-AUC | 클래스 불균형 데이터에서 보조 지표 |
| Accuracy | 참고용 |
| F1 / Precision / Recall | 임계값 기반 평가 시 |
| Log Loss | DL 모델 학습 모니터링용 |

### Regression

| 지표 | 설명 |
|------|------|
| RMSE | 기본 선택 지표 (모델 비교 기준) |
| MAE | 이상치 영향 낮은 보조 지표 |
| R² | 설명력 |
| MAPE | 상대적 오차 (0 근방 값 주의) |

## 3. 시각화 출력 (자동)

### Classification
- ROC Curve
- Precision-Recall Curve
- Confusion Matrix
- Gain / Lift Chart
- Feature Importance (SHAP summary plot)

### Regression
- Actual vs Predicted scatter plot
- Residual distribution plot
- Feature Importance (SHAP summary plot)

### 공통
- 모델별 CV score 비교 bar chart (Screening 결과)
- Stacking vs Blending 성능 비교
- Learning curve (DL 모델)

## 4. 벤치마크 평가 (Grinsztajn)

- Grinsztajn et al. (2022) 기준 데이터셋 목록 사용 (OpenML)
- 각 데이터셋에 대해 동일한 Pipeline 실행
- 논문의 Random Forest baseline 대비 성능 비교표 자동 출력

```
Dataset         | RF Baseline | Our Pipeline | Delta
----------------|-------------|--------------|-------
dataset_A       | 0.823       | 0.841        | +0.018
dataset_B       | 0.761       | 0.758        | -0.003
...
```

## 5. 출력 인터페이스

```python
report = pipe.evaluate(X_test, y_test)

report.metrics          # dict: 지표별 수치
report.plots()          # 시각화 전체 출력
report.feature_importance()  # SHAP summary plot
report.compare_models() # Screening 단계 모델별 비교
report.benchmark()      # Grinsztajn 비교표 (벤치마크 데이터셋 사용 시)
```

## 6. Explainability (XAI)

- 라이브러리: **SHAP**
- 기본 제공:
  - Global: Feature importance (mean |SHAP|)
  - Local: 단일 샘플에 대한 waterfall plot (`report.explain(X_sample)`)
- 모델별 SHAP explainer 자동 선택:
  - Tree 계열: `TreeExplainer`
  - Linear 계열: `LinearExplainer`
  - DL / 기타: `KernelExplainer` (샘플링 적용)

## 7. 확정 사항

| 항목 | 결정 |
|------|------|
| 리포트 저장 형식 | 콘솔 + 인라인 plot (기본) 및 HTML 저장 모두 지원 |
| SHAP KernelExplainer 샘플 수 | `n_samples=1000` |
