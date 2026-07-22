from __future__ import annotations
from federated_heart_disease.federated.parameters import (
    get_model_paratmeters,
)
import flwr as fl


def get_parameters(self, config):
    return get_model_paratmeters(self.model)


class HeartDiseaseClient(fl.client.NumPyClient):
    def __init__(
        self,
        hospital: str,
        X_train,
        y_train,
        X_test,
        y_test,
        model,
        pipeline,
    ):
        self.hospital = hospital

        self.X_train = X_train
        self.y_train = y_train

        self.X_test = X_test
        self.y_test = y_test

        self.model = model
        self.pipeline = pipeline

    def get_parameters(self, config):
        pass

    def fit(self, parameters, config):
        pass

    def evaluate(self, parameters, config):
        pass
