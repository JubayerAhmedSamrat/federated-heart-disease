from __future__ import annotations


from federated_heart_disease.data.scaler import create_pipeline
from federated_heart_disease.models.logistic import create_model


def train_model(X_train, y_train):

    pipeline = create_pipeline()

    X_train = pipeline.fit_transform(X_train)
    model = create_model()
    model.fit(X_train, y_train)

    return model, pipeline
