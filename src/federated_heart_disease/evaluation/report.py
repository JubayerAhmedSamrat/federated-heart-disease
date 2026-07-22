from pathlib import Path
import pandas as pd


def save_report(results: dict[str, dict[str, float]], output: Path):

    output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results).T

    df.to_csv(output)

    print(df)
