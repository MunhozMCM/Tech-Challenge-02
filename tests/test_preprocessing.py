"""Unit tests for the preprocessing stage."""

import numpy as np
import pandas as pd
import pytest

from src.config import TARGET, TEST_SIZE
from src.modeling.preprocessing import (
    encode_categoricals,
    encode_target,
    preprocess,
    scale_train_test,
    split_features_target,
)


@pytest.fixture
def toy_dataset() -> pd.DataFrame:
    """Small synthetic dataset with the column types found in the real one."""
    rng = np.random.default_rng(0)
    size = 50
    return pd.DataFrame(
        {
            "ProductRelated": rng.integers(0, 30, size),
            "PageValues": rng.random(size) * 10,
            "Month": rng.choice(["Feb", "May", "Nov"], size),
            "Weekend": rng.choice([True, False], size),
            TARGET: rng.choice([True, False], size),
        }
    )


def test_split_features_target_removes_target(toy_dataset: pd.DataFrame) -> None:
    features, target = split_features_target(toy_dataset)
    assert TARGET not in features.columns
    assert len(target) == len(toy_dataset)


def test_encode_categoricals_outputs_numeric(toy_dataset: pd.DataFrame) -> None:
    features, _ = split_features_target(toy_dataset)
    encoded = encode_categoricals(features)
    assert encoded.select_dtypes(include=["object", "bool"]).empty
    assert list(encoded.columns) == list(features.columns)


def test_encode_target_is_binary(toy_dataset: pd.DataFrame) -> None:
    encoded = encode_target(toy_dataset[TARGET])
    assert set(np.unique(encoded)) <= {0, 1}


def test_scale_train_test_fits_on_train_only(toy_dataset: pd.DataFrame) -> None:
    features, _ = split_features_target(toy_dataset)
    encoded = encode_categoricals(features)
    train, test = encoded.iloc[:35], encoded.iloc[35:]

    train_scaled, test_scaled = scale_train_test(train, test)

    np.testing.assert_allclose(train_scaled.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(train_scaled.std(axis=0), 1.0, atol=1e-9)
    # test split is transformed with train statistics, so it is NOT centered
    assert not np.allclose(test_scaled.mean(axis=0), 0.0, atol=1e-3)


def test_preprocess_end_to_end(tmp_path, toy_dataset: pd.DataFrame) -> None:
    input_path = tmp_path / "raw.csv"
    toy_dataset.to_csv(input_path, index=False)
    output_train = tmp_path / "train.csv"
    output_test = tmp_path / "test.csv"

    preprocess(str(input_path), str(output_train), str(output_test))

    train_df = pd.read_csv(output_train)
    test_df = pd.read_csv(output_test)
    assert TARGET in train_df.columns
    assert list(train_df.columns) == list(test_df.columns)
    assert len(test_df) == round(len(toy_dataset) * TEST_SIZE)
    assert len(train_df) == len(toy_dataset) - len(test_df)
