from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import config



def load_raw(dataset_path=None):
    """Load raw dataset from dataset_path or fallback to default configured in config.RAW_DATA_PATH."""
    return pd.read_csv(dataset_path or config.RAW_DATA_PATH)


def clean_data(df=None, save=True):
    data = load_raw() if df is None else df.copy()
    for column in ["Workshops", "AptitudeTestScore", "SoftSkillsRating", "CodingTestScore", "MockInterviewScore"]:
        if column in data and data[column].isna().any():
            data[column] = data[column].fillna(data[column].median())
    before = len(data)
    data = data.drop_duplicates(subset=[config.ID_COL] if config.ID_COL in data else None)
    if save:
        config.CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(config.CLEANED_DATA_PATH, index=False)
    return data, before - len(data)


def load_cleaned():
    return pd.read_csv(config.CLEANED_DATA_PATH) if config.CLEANED_DATA_PATH.exists() else clean_data()[0]


def numeric_cols(df):
    excluded = set(config.CATEGORICAL_COLS + config.TARGET_COLS + [config.ID_COL])
    return [c for c in df.select_dtypes(include="number").columns if c not in excluded]
