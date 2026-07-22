from __future__ import annotations
import pandas as pd


def validate_dataset(name: str, df: pd.DataFrame) -> None:
    print("=" * 60)
    print(f"{name.upper()}")

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nMissing Values")

    print(df.isna().sum())
    print("\nDuplicates")

    print(df.duplicated().sum())
    print("\nTarget Distribution")
    print(df["target"].value_counts().sort_index())
