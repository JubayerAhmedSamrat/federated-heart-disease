from __future__ import annotations

import pandas as pd


def preprocess(df: pd.DataFrame):
    """
    Basic preprocessing shared by all hospitals.
    """
    df = df.copy()

    # Convert '?' to NaN if any remain
    df = df.replace("?", pd.NA)

    # Convert everything to numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # Binary classifiaction
    # 0 = healthy
    # 1 = heart disease
    df["target"] = (df["target"] > 0).astype(int)

    return df
