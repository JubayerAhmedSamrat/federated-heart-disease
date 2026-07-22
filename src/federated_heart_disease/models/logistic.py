from __future__ import annotations

from sklearn.linear_model import LogisticRegression


def create_model() -> LogisticRegression:
    return LogisticRegression(
        max_iter=1000,
        random_state=42,
    )
