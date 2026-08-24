"""Central project configuration: paths, constants and MLflow settings."""

import os
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"

RAW_DATASET_PATH: Final[Path] = DATA_DIR / "online_shoppers_intention.csv"
TRAIN_DATASET_PATH: Final[Path] = DATA_DIR / "train.csv"
TEST_DATASET_PATH: Final[Path] = DATA_DIR / "test.csv"

SEED: Final[int] = 42
TEST_SIZE: Final[float] = 0.20
TARGET: Final[str] = "Revenue"

MLFLOW_TRACKING_URI: Final[str] = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
)
EXPERIMENT_NAME: Final[str] = "Propensao_Compra"
REGISTERED_MODEL_NAME: Final[str] = "RF_Propensao_Compra"

RANDOM_FOREST_PARAMS: Final[dict[str, int]] = {
    "n_estimators": 100,
    "max_depth": 5,
    "random_state": SEED,
}
