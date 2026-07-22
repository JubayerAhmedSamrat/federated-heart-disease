from __future__ import annotations

import flwr as fl


class HeartDiseaseClient(fl.client.NumPyClient):
    def __init__(self):
        pass

    def get_parameters(self, config):
        return []

    def fit(self, parameters, config):
        return [], 0, {}

    def evaluate(self, parameters, config):
        return 0.0, 0, {}
