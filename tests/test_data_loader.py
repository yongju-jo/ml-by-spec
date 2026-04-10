import numpy as np
import pandas as pd
import pytest
import sys
sys.path.insert(0, "src")

from ml_agent.data.loader import DataLoader


@pytest.fixture
def sample_data():
    X = pd.DataFrame({"a": range(100), "b": range(100, 200)})
    y = pd.Series([0, 1] * 50)
    return X, y


def test_split_shapes(sample_data):
    X, y = sample_data
    loader = DataLoader(task="classification")
    split = loader.load(X, y)

    assert split.X_train.shape[0] == 80
    assert split.X_test.shape[0] == 20
    assert len(split.y_train) == 80
    assert len(split.y_test) == 20


def test_numpy_y_converted(sample_data):
    X, y = sample_data
    loader = DataLoader(task="classification")
    split = loader.load(X, y.to_numpy())
    assert isinstance(split.y_train, pd.Series)


def test_regression_no_stratify():
    X = pd.DataFrame({"a": range(100)})
    y = pd.Series(np.random.rand(100))
    loader = DataLoader(task="regression")
    split = loader.load(X, y)
    assert split.X_train.shape[0] == 80


def test_invalid_task():
    with pytest.raises(ValueError, match="task must be"):
        DataLoader(task="clustering")


def test_length_mismatch():
    X = pd.DataFrame({"a": range(100)})
    y = pd.Series(range(50))
    loader = DataLoader(task="classification")
    with pytest.raises(ValueError, match="same length"):
        loader.load(X, y)


def test_invalid_X_type():
    loader = DataLoader(task="classification")
    with pytest.raises(TypeError):
        loader.load([[1, 2], [3, 4]], pd.Series([0, 1]))


def test_seed_reproducibility(sample_data):
    X, y = sample_data
    split1 = DataLoader(task="classification", seed=42).load(X, y)
    split2 = DataLoader(task="classification", seed=42).load(X, y)
    pd.testing.assert_frame_equal(split1.X_train, split2.X_train)
