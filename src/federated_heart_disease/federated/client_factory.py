from __future__ import annotations
from pathlib import Path

from federated_heart_disease.data.loader import load_all
from federated_heart_disease.data.preprocessing import preprocess
from federated_heart_disease.data.splitter import split_dataset
from federated_heart_disease.training.train import train_model

from federated_heart_disease.federated.client import HeartDiseaseClient


def create_client(hospital: str) -> HeartDiseaseClient:
    datasets = load_all(Path("data/raw"))
    df = preprocess(datasets[hospital])

    X_train, X_test, y_train, y_test = split_dataset(df)

    model, pipeline = train_model(
        X_train,
        y_train,
    )

    return HeartDiseaseClient(
        hospital=hospital,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model=model,
        pipeline=pipeline,
    )
