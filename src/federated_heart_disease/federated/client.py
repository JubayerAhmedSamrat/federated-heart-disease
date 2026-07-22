from __future__ import annotations
from federated_heart_disease.federated.parameters import (
    get_model_parameters,
    set_model_parameters,
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
        set_model_parameters(self.model, parameters)

        X_train = self.pipeline.transform(self.X_train)

        self.model.fit(X_train, self.y_train)

        return (
            get_model_paratmeters(self.model),
            len(self.X_train),
            {},
        )

    def evaluate(self, parameters, config):
        pass
