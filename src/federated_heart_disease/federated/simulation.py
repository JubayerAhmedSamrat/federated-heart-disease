from __future__ import annotations
from federated_heart_disease.federated.server import create_strategy


def run_federated_simulation():
    strategy = create_strategy()

    print("=" * 60)
    print("Flower Federated Learning")
    print("=" * 60)

    print(strategy)
    print("Simulatin not connected yet.")
