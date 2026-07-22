from __future__ import annotations
from federated_heart_disease.federated.parameters import (
    set_model_parameters,
)

from sklearn.metrics import accuracy_score

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
        return [
            self.model.coef_,
            self.model.intercept_,
        ]

    def fit(self, parameters, config):
        # Receive global parameters
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]

        # Transform training data
        X_train = self.pipeline.transform(self.X_train)

        # Train one local round
        self.model.fit(X_train, self.y_train)

        # Return updated parameters
        return (
            [
                self.model.coef_,
                self.model.intercept_,
            ],
            len(self.X_train),
            {},
        )

    def evaluate(self, parameters, config):
        set_model_parameters(self.model, parameters)

        X_test = self.pipeline.transform(self.X_test)

        prediction = self.model.predict(X_test)

        accuracy = accuracy_score(self.y_test, predictions)

        return (
            1.0 - accuracy,
            len(self.X_test),
            {
                "accuracy": accuracy,
            },
        )
