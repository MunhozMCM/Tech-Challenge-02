"""Training stage: fit the Random Forest and track it with MLflow."""

import argparse

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.config import (
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    RANDOM_FOREST_PARAMS,
    REGISTERED_MODEL_NAME,
    TARGET,
)
from src.data.io import read_dataset


def load_features_target(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load a preprocessed split and separate features from the target."""
    dataset = read_dataset(path)
    return dataset.drop(columns=[TARGET]), dataset[TARGET]


def train_model(features: pd.DataFrame, target: pd.Series) -> RandomForestClassifier:
    """Fit the production Random Forest configuration."""
    model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    model.fit(features, target)
    return model


def evaluate_model(
    model: RandomForestClassifier, features: pd.DataFrame, target: pd.Series
) -> dict[str, float]:
    """Compute the tracked evaluation metrics on a held-out split."""
    predictions = model.predict(features)
    return {
        "accuracy": accuracy_score(target, predictions),
        "f1_score": f1_score(target, predictions),
    }


def run_training(train_path: str, test_path: str) -> None:
    """Train, evaluate, log and register the model in a single MLflow run."""
    print("Iniciando treinamento...")
    x_train, y_train = load_features_target(train_path)
    x_test, y_test = load_features_target(test_path)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.log_params(RANDOM_FOREST_PARAMS)

        model = train_model(x_train, y_train)
        metrics = evaluate_model(model, x_test, y_test)
        mlflow.log_metrics(metrics)
        print(
            f"Modelo treinado. Acurácia: {metrics['accuracy']:.4f}, "
            f"F1-Score: {metrics['f1_score']:.4f}"
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="random_forest_model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        print("Modelo registrado no MLflow com sucesso!")


def main() -> None:
    """CLI entry point for the DVC train stage."""
    parser = argparse.ArgumentParser(description="Train the purchase-propensity model.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    args = parser.parse_args()
    run_training(args.train, args.test)


if __name__ == "__main__":
    main()
