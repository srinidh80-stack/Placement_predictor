from pathlib import Path

import pandas as pd


DATA_FILE = Path(__file__).resolve().parents[3] / "Data" / "Raw Data" / "placement_predict_50k Dataset (2).csv"


def load_data(dataset_path=None):
    return pd.read_csv(dataset_path or DATA_FILE)


def summarize(df):
    return {
        "shape": df.shape,
        "columns": df.columns,
    }
