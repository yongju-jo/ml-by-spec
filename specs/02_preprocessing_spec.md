# 02. Preprocessing Spec

## 1. 목표

데이터 타입을 자동으로 감지하고, 타입별로 적절한 전처리를 자동 적용한다.
사용자는 별도 설정 없이 raw DataFrame을 넣으면 된다.

## 2. 컬럼 타입 자동 감지

| 감지 타입 | 판단 기준 |
|-----------|-----------|
| `numeric` | int / float dtype, 고유값 비율 높음 |
| `categorical` | object / string dtype, 또는 int이지만 고유값 수 적음 (기본: ≤ 20) |
| `binary` | 고유값이 2개인 컬럼 |
| `datetime` | datetime dtype 또는 날짜 패턴 문자열 |
| `text` | 평균 토큰 수가 일정 기준 이상인 string 컬럼 (Out of Scope) |

## 3. 타입별 전처리 파이프라인

### 3.1 Numeric
1. 결측치 처리: Median Imputation (기본) / Mean / Constant Override 가능
2. 이상치 처리: IQR Capping (선택, 기본 Off)
3. 스케일링:
   - Tree 계열 모델: 스케일링 없음
   - Linear / DL 계열 모델: StandardScaler (기본) / MinMaxScaler / RobustScaler

### 3.2 Categorical
1. 결측치 처리: 최빈값 Imputation 또는 `"missing"` 카테고리 추가
2. 인코딩:
   - 고유값 수 적음 (≤ 기본 10): OneHotEncoding
   - 고유값 수 많음 (> 기본 10): OrdinalEncoding (Tree 계열) / TargetEncoding (DL 계열)
   - CatBoost: 자체 내부 처리 위임

### 3.3 Binary
1. 결측치 처리: 최빈값 Imputation
2. 인코딩: Label Encoding (0/1)

### 3.4 Datetime
1. 분해: year, month, day, weekday, hour 파생 변수 생성
2. 원본 컬럼 제거

## 4. 전처리 순서

```
Raw DataFrame
    ↓
타입 감지 (TypeDetector)
    ↓
결측치 처리 (Imputer)
    ↓
인코딩 (Encoder)
    ↓
스케일링 (Scaler) — 모델 타입에 따라 분기
    ↓
Processed DataFrame
```

## 5. 모델 계열별 전처리 분기

전처리는 모델 계열에 따라 다르게 적용된다.
Pipeline 내부에서 모델별로 별도의 Preprocessor 인스턴스를 유지한다.

| 모델 계열 | 스케일링 | 인코딩 |
|-----------|----------|--------|
| Tree (RF, ExtraTree) | 없음 | OrdinalEncoding |
| Boosting (XGB, LGB) | 없음 | OrdinalEncoding |
| CatBoost | 없음 | 자체 처리 |
| Linear (LR, Ridge) | StandardScaler | OneHotEncoding |
| DL (MLP, TabNet, FT-Transformer) | StandardScaler | TargetEncoding / OneHotEncoding |

## 6. Override 인터페이스

```python
from ml_agent.preprocessing import PreprocessorConfig

config = PreprocessorConfig(
    numeric_imputer="mean",
    scaler="robust",
    categorical_threshold=15,       # OHE vs OrdinalEncoding 경계 고유값 수
    outlier_capping=True,
)

pipe = Pipeline(task="classification", preprocessor_config=config)
```

## 7. 확정 사항

| 항목 | 결정 |
|------|------|
| TargetEncoding leakage 방지 | sklearn `TargetEncoder` 그대로 사용 |
| 고차원 범주형 처리 | 빈도 상위 5개 카테고리 유지, 나머지 `"other"`로 대체 후 OHE/OrdinalEncoding 적용 |
| datetime 파생 변수 | year, month, day, weekday, hour (고정) |
