from __future__ import annotations
from flwr.simulation import run_simulation
from federated_heart_disease.federated.server import create_strategy


def run_federated_simulation():
    strategy = create_strategy()

    print(strategy)
    print(run_simulation)
    print("Flower simulation comming soon...")
