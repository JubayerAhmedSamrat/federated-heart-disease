from __future__ import annotations
from federated_heart_disease.federated.server import create_strategy
from federated_heart_disease.federated.client_factory import create_client

HOSPITALS = ["cleveland", "hungarian", "switzerland", "va"]


def run_federated_simulation():
    strategy = create_strategy()

    print("=" * 60)
    print("Flower Federated Learning")
    print("=" * 60)

    print("Strategy:")
    print(strategy)

    print()

    print("Clients:")

    for hospital in HOSPITALS:
        client = create_client(hospital)
        print(f"✓ {hospital} -> {client}")


if __name__ == "__main__":
    run_federated_simulation()
