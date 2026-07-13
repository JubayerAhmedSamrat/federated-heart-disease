from pathlib import Path 
from federated_heart_disease.data.downloader import download_dataset

def main() -> None:
    download_dataset(Path("data/raw"))

if __name__ == "__main__":
    main()
