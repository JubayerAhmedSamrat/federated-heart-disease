from __future__ import annotations
from pathlib import Path
import pandas as pd

COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]

FILES = {
    "cleveland": "processed.cleveland.data",
    "hungarian": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va": "processed.va.data",
}


def load_dataset(path: Path) -> pd.DataFrame:
    """Load one heart disease dataset."""

    return pd.read_csv(
        path,
        header=None,
        names=COLUMN_NAMES,
        na_values="?",
    )


def load_all(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all hospital datasets."""

    datasets: dict[str, pd.DataFrame] = {}
    for hospital, filename in FILES.items():
        datasets[hospital] = load_dataset(data_dir / filename)

    return datasets
