import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import pytest

from ml_agent.data.type_detector import TypeDetector
from ml_agent.preprocessing.preprocessor import AutoPreprocessor
from ml_agent.preprocessing.config import PreprocessorConfig


@pytest.fixture
def sample_df():
    np.random.seed(42)
    return pd.DataFrame({
        "age":       np.random.randint(20, 60, 100).astype(float),
        "income":    np.random.rand(100) * 100000,
        "city":      np.random.choice(["Seoul", "Busan", "Daegu"], 100),
        "gender":    np.random.choice(["M", "F"], 100),
        "signup":    pd.date_range("2020-01-01", periods=100, freq="7D"),
    })


@pytest.fixture
def sample_y():
    np.random.seed(42)
    return pd.Series(np.random.randint(0, 2, 100), name="target")


@pytest.fixture
def type_map(sample_df):
    return TypeDetector().detect(sample_df)


def test_tree_no_scaling(sample_df, sample_y, type_map):
    prep = AutoPreprocessor()
    result = prep.fit_transform(sample_df, sample_y, type_map, model_family="tree")
    assert isinstance(result, pd.DataFrame)
    # age, income은 스케일링 없으므로 원본 범위 유지
    assert result.shape[0] == 100


def test_linear_scaling(sample_df, sample_y, type_map):
    prep = AutoPreprocessor()
    result = prep.fit_transform(sample_df, sample_y, type_map, model_family="linear")
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 100


def test_datetime_decomposed(sample_df, sample_y, type_map):
    prep = AutoPreprocessor()
    result = prep.fit_transform(sample_df, sample_y, type_map, model_family="tree")
    cols = result.columns.tolist()
    # signup 컬럼이 파생 변수로 분해되어야 함
    assert any("signup" in c for c in cols)
    assert "signup" not in cols  # 원본 컬럼은 제거


def test_binary_encoded(sample_df, sample_y, type_map):
    prep = AutoPreprocessor()
    result = prep.fit_transform(sample_df, sample_y, type_map, model_family="tree")
    # gender는 binary → 0/1로 인코딩
    assert result.shape[0] == 100


def test_transform_consistency(sample_df, sample_y, type_map):
    prep = AutoPreprocessor()
    train = prep.fit_transform(sample_df, sample_y, type_map, model_family="tree")
    test = prep.transform(sample_df)
    assert train.shape == test.shape


def test_transform_before_fit_raises():
    prep = AutoPreprocessor()
    with pytest.raises(RuntimeError, match="not fitted"):
        prep.transform(pd.DataFrame({"a": [1, 2]}))


def test_missing_values_handled(sample_y, type_map):
    df = pd.DataFrame({
        "age":    [25.0, np.nan, 30.0, np.nan, 35.0] * 20,
        "city":   ["Seoul", None, "Busan", "Daegu", None] * 20,
        "gender": ["M", "F", None, "M", "F"] * 20,
        "signup": pd.date_range("2020-01-01", periods=100, freq="7D"),
    })
    tm = TypeDetector().detect(df)
    prep = AutoPreprocessor()
    result = prep.fit_transform(df, sample_y, tm, model_family="tree")
    assert not result.isnull().any().any()


def test_high_cardinality_top_k():
    # 고유값 많은 범주형 컬럼 → 상위 5개 + other로 축소
    np.random.seed(0)
    df = pd.DataFrame({
        "product_id": [f"P{i}" for i in np.random.randint(0, 50, 100)],
        "value": np.random.rand(100),
    })
    y = pd.Series(np.random.randint(0, 2, 100))
    tm = TypeDetector(categorical_threshold=5).detect(df)
    config = PreprocessorConfig(ohe_threshold=5, high_cardinality_top_k=5)
    prep = AutoPreprocessor(config=config)
    result = prep.fit_transform(df, y, tm, model_family="tree")
    assert result.shape[0] == 100


def test_outlier_capping(sample_y, type_map):
    df = pd.DataFrame({
        "age":    [25.0] * 95 + [9999.0, -9999.0, 1e6, -1e6, 500.0],
        "income": np.random.rand(100) * 100000,
        "city":   ["Seoul"] * 100,
        "gender": ["M", "F"] * 50,
        "signup": pd.date_range("2020-01-01", periods=100, freq="7D"),
    })
    tm = TypeDetector().detect(df)
    config = PreprocessorConfig(outlier_capping=True)
    prep = AutoPreprocessor(config=config)
    result = prep.fit_transform(df, sample_y, tm, model_family="tree")
    assert result.shape[0] == 100
