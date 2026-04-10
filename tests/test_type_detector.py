import pandas as pd
import pytest
import sys
sys.path.insert(0, "src")

from ml_agent.data.type_detector import TypeDetector, NUMERIC, CATEGORICAL, BINARY, DATETIME


@pytest.fixture
def detector():
    return TypeDetector(categorical_threshold=20)


def test_numeric_column(detector):
    X = pd.DataFrame({"price": [float(i) + 0.1 for i in range(50)]})
    result = detector.detect(X)
    assert "price" in result.numeric


def test_categorical_column(detector):
    X = pd.DataFrame({"city": ["Seoul", "Busan", "Daegu"] * 10})
    result = detector.detect(X)
    assert "city" in result.categorical


def test_binary_column(detector):
    X = pd.DataFrame({"is_churn": [0, 1, 0, 1, 0]})
    result = detector.detect(X)
    assert "is_churn" in result.binary


def test_datetime_dtype(detector):
    X = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"])})
    result = detector.detect(X)
    assert "date" in result.datetime


def test_datetime_string_pattern(detector):
    X = pd.DataFrame({"date": ["2024-01-01", "2024-02-15", "2024-03-20"] * 10})
    result = detector.detect(X)
    assert "date" in result.datetime


def test_int_low_cardinality_as_categorical(detector):
    # 고유값 수 <= threshold → categorical
    X = pd.DataFrame({"grade": [1, 2, 3, 4, 5] * 4})
    result = detector.detect(X)
    assert "grade" in result.categorical


def test_int_high_cardinality_as_numeric(detector):
    X = pd.DataFrame({"amount": range(100)})
    result = detector.detect(X)
    assert "amount" in result.numeric


def test_mixed_columns(detector):
    X = pd.DataFrame({
        "age": range(50),
        "gender": ["M", "F"] * 25,
        "score": [1.5, 2.5, 3.5] * 16 + [1.5, 2.5],
        "signup_date": pd.date_range("2020-01-01", periods=50),
    })
    result = detector.detect(X)
    assert "age" in result.numeric
    assert "gender" in result.binary
    assert "signup_date" in result.datetime
