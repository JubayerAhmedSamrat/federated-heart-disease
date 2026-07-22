from __future__ import annotations

from flwr.server.strategy import FedAvg


def create_strategy():
    return FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_avaiable_clients=4,
    )
