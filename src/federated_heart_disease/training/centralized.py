from __future__ import annotations

from pathlib import Path

from federated_heart_disease.data.loader import load_all
from federated_heart_disease.data.preprocessing import preprocess
from federated_heart_disease.data.splitter import split_dataset
from federated_heart_disease.training.train import train_model
from federated_heart_disease.evaluation.metrics import evaluate
from federated_heart_disease.evaluation.report import save_report
from federated_heart_disease.utils.io import save_model


def run_centralized_training(data_dir: Path):
    datasets = load_all(data_dir)

    results = {}

    for hospitals, df in datasets.items():
        df = preprocess(df)

        X_train, X_test, y_train, y_test = split_dataset(df)

        model, pipeline = train_model(X_train, y_train)

        save_model(
            model,
            Path(f"results/models/{hospitals}_logistic.pk1"),
        )

        metrics = evaluate(
            model,
            pipeline,
            X_test,
            y_test,
        )
        results[hospitals] = metrics

        print("=" * 60)
        print(hospitals)

        for key, value in metrics.items():
            print(f"{key:10}: {value:.4f}")

            save_report(
                results,
                Path("results/metrics.csv"),
            )

            return results
