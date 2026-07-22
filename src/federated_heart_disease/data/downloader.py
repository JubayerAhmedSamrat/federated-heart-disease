from __future__ import annotations

from pathlib import Path
import requests

BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease"

FILES = {
    "cleveland": "processed.cleveland.data",
    "hungarian": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va": "processed.va.data",
}


def download_dataset(output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    destination = output_dir / filename

    if destination.exists():
        print(f"{filename} already exists")
        return

    url = f"{BASE_URL}/{filename}"

    print(f"Downloading {filename}...")

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    destination.write_bytes(response.content)

    print(f"Saved {destination}")


def download_all(output_dir: Path) -> None:
    for filename in FILES.values():
        download_dataset(output_dir, filename)
