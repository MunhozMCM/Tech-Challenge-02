"""Preprocessing stage: encode, split and scale the shoppers dataset.

Behavior mirrors the original pipeline (label encoding of categorical
features); the one-hot upgrade is documented as a next step in
notebooks/ML_experiments_decisions.md.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import SEED, TARGET, TEST_SIZE
from src.data.io import read_dataset, save_dataset


def split_features_target(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the feature matrix from the target column."""
    return dataset.drop(columns=[TARGET]), dataset[TARGET]


def encode_categoricals(features: pd.DataFrame) -> pd.DataFrame:
    """Label-encode object/bool feature columns into integer codes."""
    encoded = features.copy()
    for column in encoded.select_dtypes(include=["object", "bool"]).columns:
        encoded[column] = LabelEncoder().fit_transform(encoded[column])
    return encoded


def encode_target(target: pd.Series) -> np.ndarray:
    """Encode the boolean target into 0/1 integers."""
    return LabelEncoder().fit_transform(target)


def scale_train_test(
    train_features: pd.DataFrame, test_features: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize features, fitting the scaler on the training split only."""
    scaler = StandardScaler()
    return scaler.fit_transform(train_features), scaler.transform(test_features)


def _assemble(features: np.ndarray, target: np.ndarray, columns: pd.Index) -> pd.DataFrame:
    """Rebuild a labelled DataFrame from scaled features plus the target."""
    dataset = pd.DataFrame(features, columns=columns)
    dataset[TARGET] = target
    return dataset


def preprocess(input_path: str, output_train: str, output_test: str) -> None:
    """Run the full preprocessing stage: load, encode, split, scale and save."""
    print(f"Carregando dados de {input_path}...")
    dataset = read_dataset(input_path)

    features, target = split_features_target(dataset)
    features = encode_categoricals(features)
    encoded_target = encode_target(target)

    x_train, x_test, y_train, y_test = train_test_split(
        features, encoded_target, test_size=TEST_SIZE, random_state=SEED
    )
    x_train_scaled, x_test_scaled = scale_train_test(x_train, x_test)

    save_dataset(_assemble(x_train_scaled, y_train, features.columns), output_train)
    save_dataset(_assemble(x_test_scaled, y_test, features.columns), output_test)
    print(f"Dados salvos em {output_train} e {output_test}.")


def main() -> None:
    """CLI entry point for the DVC preprocess stage."""
    parser = argparse.ArgumentParser(description="Preprocess the shoppers dataset.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-train", required=True)
    parser.add_argument("--output-test", required=True)
    args = parser.parse_args()
    preprocess(args.input, args.output_train, args.output_test)


if __name__ == "__main__":
    main()
