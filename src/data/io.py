"""Dataset input/output helpers."""

from pathlib import Path

import pandas as pd


def read_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset into a DataFrame."""
    return pd.read_csv(path)


def save_dataset(dataset: pd.DataFrame, path: str | Path) -> None:
    """Write a DataFrame to CSV (without the index), creating parent dirs if needed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(destination, index=False)
