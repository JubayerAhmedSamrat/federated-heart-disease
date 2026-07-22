from pathlib import Path
from federated_heart_disease.data.loader import load_all


def main() -> None:
    datasets = load_all(Path("data/raw"))

    for name, df in datasets.items():
        print("=" * 50)
        print(name)
        print(df.shape)
        print(df.head())


if __name__ == "__main__":
    main()
