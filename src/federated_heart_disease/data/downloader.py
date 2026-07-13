from __future__ import annotations
from pathlib import Path  
from ucimlrepo import fetch_ucirepo

DATASET_ID = 45

def download_dataset(output_dir: Path) -> None:
    """
    Download the UCI Heart Disease dataset.

    Parameters
    -----------
    output_dir: Path 
        Directory where raw dataset files will be stored
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = fetch_ucirepo(id=DATASET_ID)

    print(dataset.metadata)
    print(dataset.variables)
