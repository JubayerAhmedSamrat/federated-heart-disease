from pathlib import Path

from federated_heart_disease.data.downloader import download_all
from federated_heart_disease.training.centralized import (
    run_centralized_training,
)


def main():
    download_all(Path("data/raw"))

    run_centralized_training(Path("data/raw"))


if __name__ == "__main__":
    main()
