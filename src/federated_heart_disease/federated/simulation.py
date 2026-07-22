from __future__ import annotations
from flwr.simulation import run_simulation
from federated_heart_disease.federated.server import create_strategy


def main():
    strategy = create_strategy()

    print("Flower simulation is ready.")
    print(strategy)


if __name__ == "__main__":
    main()
