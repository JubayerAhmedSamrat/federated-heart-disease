from __future__ import annotations

from federated_heart_disease.federated.parameters import (
    get_model_parameters,
    set_model_parameters,
)

import flwr as fl


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
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        # Receive global parameters
        set_model_parameters(self.model, parameters)
        # Transform training data
        X_train = self.pipeline.transform(self.X_train)

        # Train one local round
        self.model.fit(X_train, self.y_train)

        # Return updated parameters
        return (
            get_model_parameters(self.model),
            len(self.X_train),
            {},
        )

    def evaluate(self, parameters, config):
        # Update model with global parameters
        set_model_parameters(self.model, parameters)
        # Transform test data
        X_test = self.pipeline.transform(self.X_test)

        # Evaluate
        accuracy = self.model.score(X_test, self.y_test)
        loss = 1.0 - accuracy

        return (
            loss,
            len(self.X_test),
            {
                "accuracy": accuracy,
            },
        )
