"""Feature scaling and encoding helpers for the placement prediction dataset.

The formulas mirror the session PPT:
- Min-Max: x' = (x - x_min) / (x_max - x_min)
- Standard/Z-score: z = (x - mean) / std
- Robust: x' = (x - median) / IQR
- One-hot encoding: create one binary column for each category
- Ordinal encoding: replace ordered categories with integer ranks
- Target encoding: replace a category with the smoothed mean target value
- Embedding-based encoding: learned dense vectors for deep learning models
- Missing-value handling: deletion, imputation, model-based imputation, and indicators

Scaler statistics are learned from training data only and then reused for
test/serving data to avoid data leakage. Encoders follow the same rule by
learning category names/orders or target means from the training data only.
"""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import load_cleaned, numeric_cols
import config


ONE_HOT_COLUMNS = ["Gender", "City", "Stream", "Specialisation"]

TARGET_ENCODING_COLUMNS = ["City"]

ORDINAL_CATEGORY_ORDERS = {
    "CollegeTier": ["Tier3", "Tier2", "Tier1"],
    "Hostel": ["No", "Yes"],
    "HistoryOfBacklogs": ["No", "Yes"],
    "CGPA_Tier": ["Low", "Mid", "High"],
    "ExtraCurricular": [0, 1],
}


FEATURE_ENGINEERING_CONCEPTS = [
    ("Min-Max Scaling", "x' = (x - x_min) / (x_max - x_min); keeps values in a fixed range."),
    ("Standard Scaling", "z = (x - mean) / std; centers features around zero."),
    ("Robust Scaling", "x' = (x - median) / IQR; reduces the effect of strong outliers."),
    ("One-Hot Encoding", "Creates one binary column per category for nominal, low-cardinality features."),
    ("Ordinal Encoding", "Uses integer ranks only when the category has a real order."),
    (
        "Target Encoding",
        "Replaces each category with mean(target | category); fit on train only and smooth rare categories.",
    ),
    (
        "Embedding-Based Encoding",
        "Learns a dense vector per category inside a neural network; useful for very high cardinality.",
    ),
]


MISSING_VALUE_CONCEPTS = [
    (
        "Missingness Mechanisms",
        "Diagnose MCAR, MAR, MNAR, or structural missingness before choosing a fix.",
    ),
    (
        "Listwise Row Deletion",
        "Drops rows containing missing model features; simple, but can discard too much data.",
    ),
    (
        "Column Deletion",
        "Drops a feature only when missingness is high and the column has low predictive value.",
    ),
    (
        "Mean / Median Imputation",
        "Fills numeric gaps with a train-fitted statistic; median is safer for skewed or outlier-heavy columns.",
    ),
    (
        "Model-Based Imputation",
        "Uses KNN or iterative models to predict missing values from other observed features.",
    ),
    (
        "Missing-Indicator Features",
        "Adds binary was-missing columns so the model can learn whether a value was imputed.",
    ),
    (
        "Leak-Free Pipeline Rule",
        "Split first, then fit imputers on training data only and transform validation/test data.",
    ),
]


def scaling_columns(df, columns=None):
    """Return numeric feature columns, excluding ids, targets, and categoricals."""
    if columns is not None:
        missing = [column for column in columns if column not in df.columns]
        if missing:
            raise KeyError(f"Columns not found: {missing}")
        return list(columns)
    return numeric_cols(df)


def categorical_columns(df, columns=None):
    """Return available categorical feature columns, excluding ids and targets."""
    if columns is not None:
        missing = [column for column in columns if column not in df.columns]
        if missing:
            raise KeyError(f"Columns not found: {missing}")
        return list(columns)

    excluded = set(config.TARGET_COLS + [config.ID_COL])
    return [column for column in config.CATEGORICAL_COLS if column in df.columns and column not in excluded]


def train_test_split_df(df, test_size=0.2, random_state=42, stratify_col=None):
    """Small pandas train/test splitter used to keep this module dependency-free."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    data = df.copy()

    if stratify_col and stratify_col in data.columns:
        test_indices = []
        for _, group in data.groupby(stratify_col, dropna=False):
            test_count = max(1, int(round(len(group) * test_size)))
            sampled = group.sample(n=test_count, random_state=random_state)
            test_indices.extend(sampled.index)
        test_indices = pd.Index(test_indices)
    else:
        test_count = int(round(len(data) * test_size))
        test_indices = data.sample(n=test_count, random_state=random_state).index

    train_df = data.drop(index=test_indices).reset_index(drop=True)
    test_df = data.loc[test_indices].reset_index(drop=True)
    return train_df, test_df


def _safe_denominator(values):
    return values.replace(0, 1)


def fit_one_hot_encoder(train_df, columns=None, drop_first=False, dummy_na=False):
    """Learn category values from training data for one-hot encoding."""
    default_cols = [column for column in ONE_HOT_COLUMNS if column in train_df.columns]
    cols = categorical_columns(train_df, columns if columns is not None else default_cols)
    categories = {}
    encoded_columns = []

    for column in cols:
        values = train_df[column]
        column_categories = sorted(values.dropna().unique().tolist())
        if drop_first and column_categories:
            column_categories = column_categories[1:]
        categories[column] = column_categories

        dummy_source = pd.Categorical(values, categories=categories[column])
        dummy_cols = pd.get_dummies(dummy_source, prefix=column, dummy_na=dummy_na).columns.tolist()
        encoded_columns.extend(dummy_cols)

    return {
        "method": "one_hot",
        "columns": cols,
        "categories": categories,
        "encoded_columns": encoded_columns,
        "drop_first": drop_first,
        "dummy_na": dummy_na,
    }


def transform_one_hot(df, encoder):
    """Apply one-hot encoding with training categories and stable output columns."""
    output = df.copy()
    encoded_parts = []

    for column in encoder["columns"]:
        categories = encoder["categories"][column]
        values = pd.Categorical(output[column], categories=categories)
        dummies = pd.get_dummies(values, prefix=column, dummy_na=encoder["dummy_na"])
        encoded_parts.append(dummies)

    output = output.drop(columns=encoder["columns"])
    if not encoded_parts:
        return output

    encoded = pd.concat(encoded_parts, axis=1)
    encoded = encoded.reindex(columns=encoder["encoded_columns"], fill_value=0)
    return pd.concat([output, encoded], axis=1)


def one_hot_encode(train_df, test_df=None, columns=None, drop_first=False, dummy_na=False):
    """Fit one-hot encoder on train_df and transform train_df/test_df."""
    encoder = fit_one_hot_encoder(train_df, columns, drop_first, dummy_na)
    train_encoded = transform_one_hot(train_df, encoder)
    test_encoded = transform_one_hot(test_df, encoder) if test_df is not None else None
    return train_encoded, test_encoded, encoder


def fit_ordinal_encoder(train_df, columns=None, category_orders=None, unknown_value=-1):
    """Learn ordinal mappings from ordered categories."""
    category_orders = category_orders or ORDINAL_CATEGORY_ORDERS
    default_cols = [column for column in category_orders if column in train_df.columns]
    cols = categorical_columns(train_df, columns if columns is not None else default_cols)
    mappings = {}

    for column in cols:
        if column in category_orders:
            ordered_values = category_orders[column]
        else:
            ordered_values = train_df[column].dropna().drop_duplicates().tolist()
        mappings[column] = {value: index for index, value in enumerate(ordered_values)}

    return {
        "method": "ordinal",
        "columns": cols,
        "mappings": mappings,
        "unknown_value": unknown_value,
    }


def transform_ordinal(df, encoder):
    """Apply ordinal mappings and use unknown_value for unseen categories."""
    output = df.copy()
    for column in encoder["columns"]:
        output[column] = output[column].map(encoder["mappings"][column]).fillna(encoder["unknown_value"]).astype(int)
    return output


def ordinal_encode(train_df, test_df=None, columns=None, category_orders=None, unknown_value=-1):
    """Fit ordinal encoder on train_df and transform train_df/test_df."""
    encoder = fit_ordinal_encoder(train_df, columns, category_orders, unknown_value)
    train_encoded = transform_ordinal(train_df, encoder)
    test_encoded = transform_ordinal(test_df, encoder) if test_df is not None else None
    return train_encoded, test_encoded, encoder


def fit_target_encoder(train_df, columns=None, target_col="PlacementStatus", smoothing=50):
    """Learn smoothed target means from training data for target encoding.

    Formula from the PPT:
    smoothed = (n_c * mean_c + m * global_mean) / (n_c + m)
    """
    if target_col not in train_df.columns:
        raise KeyError(f"Target column not found: {target_col}")
    if smoothing < 0:
        raise ValueError("smoothing must be greater than or equal to 0")

    default_cols = [column for column in TARGET_ENCODING_COLUMNS if column in train_df.columns]
    cols = categorical_columns(train_df, columns if columns is not None else default_cols)
    global_mean = train_df[target_col].mean()
    mappings = {}

    for column in cols:
        grouped = train_df.groupby(column, dropna=False)[target_col].agg(["mean", "count"])
        smoothed = (grouped["count"] * grouped["mean"] + smoothing * global_mean) / (
            grouped["count"] + smoothing
        )
        mappings[column] = smoothed

    return {
        "method": "target",
        "columns": cols,
        "target_col": target_col,
        "global_mean": global_mean,
        "mappings": mappings,
        "smoothing": smoothing,
    }


def transform_target(df, encoder, drop_original=True, suffix="_target_enc"):
    """Apply target encoding and use global mean for unseen categories."""
    output = df.copy()
    encoded_columns = []

    for column in encoder["columns"]:
        encoded_column = f"{column}{suffix}"
        output[encoded_column] = output[column].map(encoder["mappings"][column])
        output[encoded_column] = output[encoded_column].fillna(encoder["global_mean"])
        encoded_columns.append(encoded_column)

    if drop_original and encoder["columns"]:
        output = output.drop(columns=encoder["columns"])

    return output, encoded_columns


def _fold_indices(index, n_splits=5, random_state=42):
    """Return deterministic validation folds without adding a sklearn dependency."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if len(index) < n_splits:
        raise ValueError("n_splits cannot be larger than the number of training rows")

    shuffled = pd.Series(list(index)).sample(frac=1, random_state=random_state).tolist()
    return [shuffled[index::n_splits] for index in range(n_splits)]


def target_encode(
    train_df,
    test_df=None,
    columns=None,
    target_col="PlacementStatus",
    smoothing=50,
    out_of_fold=True,
    n_splits=5,
    random_state=42,
    drop_original=True,
    suffix="_target_enc",
):
    """Fit target encoder on train_df and transform train_df/test_df.

    The test set always uses mappings fitted on the full training set. By default,
    the training set uses out-of-fold mappings so each row is encoded from other
    rows only, reducing target leakage from in-fold means.
    """
    encoder = fit_target_encoder(train_df, columns, target_col, smoothing)
    test_encoded = None
    if test_df is not None:
        test_encoded = transform_target(test_df, encoder, drop_original, suffix)[0]

    if not out_of_fold or len(train_df) < n_splits:
        train_encoded = transform_target(train_df, encoder, drop_original, suffix)[0]
        return train_encoded, test_encoded, encoder

    train_encoded = train_df.copy()
    encoded_columns = [f"{column}{suffix}" for column in encoder["columns"]]
    for encoded_column in encoded_columns:
        train_encoded[encoded_column] = encoder["global_mean"]

    for fold in _fold_indices(train_df.index, n_splits, random_state):
        fit_part = train_df.drop(index=fold)
        fold_encoder = fit_target_encoder(fit_part, encoder["columns"], target_col, smoothing)
        fold_encoded = transform_target(train_df.loc[fold], fold_encoder, False, suffix)[0]
        for encoded_column in encoded_columns:
            train_encoded.loc[fold, encoded_column] = fold_encoded[encoded_column]

    if drop_original and encoder["columns"]:
        train_encoded = train_encoded.drop(columns=encoder["columns"])

    encoder["out_of_fold"] = out_of_fold
    encoder["n_splits"] = n_splits
    return train_encoded, test_encoded, encoder


def fit_min_max_scaler(train_df, columns=None, feature_range=(0, 1)):
    """Learn min and max from training data for Min-Max normalization."""
    cols = scaling_columns(train_df, columns)
    data_min = train_df[cols].min()
    data_max = train_df[cols].max()
    return {
        "method": "min_max",
        "columns": cols,
        "min": data_min,
        "max": data_max,
        "feature_range": feature_range,
    }


def transform_min_max(df, scaler):
    """Apply Min-Max normalization with already learned training statistics."""
    output = df.copy()
    cols = scaler["columns"]
    low, high = scaler["feature_range"]
    denominator = _safe_denominator(scaler["max"] - scaler["min"])
    output[cols] = (output[cols] - scaler["min"]) / denominator
    output[cols] = output[cols] * (high - low) + low
    return output


def min_max_scale(train_df, test_df=None, columns=None, feature_range=(0, 1)):
    """Fit Min-Max on train_df and transform train_df/test_df."""
    scaler = fit_min_max_scaler(train_df, columns, feature_range)
    train_scaled = transform_min_max(train_df, scaler)
    test_scaled = transform_min_max(test_df, scaler) if test_df is not None else None
    return train_scaled, test_scaled, scaler


def fit_standard_scaler(train_df, columns=None):
    """Learn mean and population std from training data for Z-score scaling."""
    cols = scaling_columns(train_df, columns)
    mean = train_df[cols].mean()
    std = train_df[cols].std(ddof=0)
    return {
        "method": "standard",
        "columns": cols,
        "mean": mean,
        "std": std,
    }


def transform_standard(df, scaler):
    """Apply Z-score standardization with training mean and std."""
    output = df.copy()
    cols = scaler["columns"]
    denominator = _safe_denominator(scaler["std"])
    output[cols] = (output[cols] - scaler["mean"]) / denominator
    return output


def standard_scale(train_df, test_df=None, columns=None):
    """Fit Standard/Z-score scaler on train_df and transform train_df/test_df."""
    scaler = fit_standard_scaler(train_df, columns)
    train_scaled = transform_standard(train_df, scaler)
    test_scaled = transform_standard(test_df, scaler) if test_df is not None else None
    return train_scaled, test_scaled, scaler


def fit_robust_scaler(train_df, columns=None):
    """Learn median and IQR from training data for Robust scaling."""
    cols = scaling_columns(train_df, columns)
    q1 = train_df[cols].quantile(0.25)
    q3 = train_df[cols].quantile(0.75)
    return {
        "method": "robust",
        "columns": cols,
        "median": train_df[cols].median(),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
    }


def transform_robust(df, scaler):
    """Apply Robust scaling with training median and IQR."""
    output = df.copy()
    cols = scaler["columns"]
    denominator = _safe_denominator(scaler["iqr"])
    output[cols] = (output[cols] - scaler["median"]) / denominator
    return output


def robust_scale(train_df, test_df=None, columns=None):
    """Fit Robust scaler on train_df and transform train_df/test_df."""
    scaler = fit_robust_scaler(train_df, columns)
    train_scaled = transform_robust(train_df, scaler)
    test_scaled = transform_robust(test_df, scaler) if test_df is not None else None
    return train_scaled, test_scaled, scaler


def clip_tukey_outliers(train_df, test_df=None, columns=None, factor=1.5):
    """Clip strong outliers using Tukey fences before applying sensitive scalers."""
    cols = scaling_columns(train_df, columns)
    q1 = train_df[cols].quantile(0.25)
    q3 = train_df[cols].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr

    train_clipped = train_df.copy()
    train_clipped[cols] = train_clipped[cols].clip(lower=lower, upper=upper, axis=1)

    test_clipped = None
    if test_df is not None:
        test_clipped = test_df.copy()
        test_clipped[cols] = test_clipped[cols].clip(lower=lower, upper=upper, axis=1)

    return train_clipped, test_clipped, {"columns": cols, "lower": lower, "upper": upper}


def build_scaled_splits(method="standard", df=None, columns=None, test_size=0.2, random_state=42):
    """Load/split data and apply one of: min_max, standard, robust."""
    data = load_cleaned() if df is None else df.copy()
    stratify = "PlacementStatus" if "PlacementStatus" in data.columns else None
    train_df, test_df = train_test_split_df(data, test_size, random_state, stratify)

    scalers = {
        "min_max": min_max_scale,
        "standard": standard_scale,
        "robust": robust_scale,
    }
    if method not in scalers:
        raise ValueError(f"Unknown scaling method: {method}. Choose from {list(scalers)}")

    return scalers[method](train_df, test_df, columns)


def build_encoded_splits(method="one_hot", df=None, columns=None, test_size=0.2, random_state=42, **kwargs):
    """Load/split data and apply one of: one_hot, ordinal, target."""
    data = load_cleaned() if df is None else df.copy()
    stratify = "PlacementStatus" if "PlacementStatus" in data.columns else None
    train_df, test_df = train_test_split_df(data, test_size, random_state, stratify)

    encoders = {
        "one_hot": one_hot_encode,
        "ordinal": ordinal_encode,
        "target": target_encode,
    }
    if method not in encoders:
        raise ValueError(f"Unknown encoding method: {method}. Choose from {list(encoders)}")

    return encoders[method](train_df, test_df, columns, **kwargs)


if __name__ == "__main__":
    cols = [
        "CGPA",
        "AttendancePercent",
        "Internships",
        "Projects",
        "Workshops",
        "Certifications",
        "AptitudeTestScore",
        "SoftSkillsRating",
        "CodingTestScore",
        "MockInterviewScore",
    ]
    data = load_cleaned()
    cols = [column for column in cols if column in data.columns]

    print("Handling missing values concepts:")
    for index, (name, description) in enumerate(MISSING_VALUE_CONCEPTS, start=1):
        print(f"{index}. {name}: {description}")

    print("Feature engineering concepts:")
    for index, (name, description) in enumerate(FEATURE_ENGINEERING_CONCEPTS, start=1):
        print(f"{index}. {name}: {description}")

    for scaler_name in ["min_max", "standard", "robust"]:
        train_scaled, test_scaled, scaler = build_scaled_splits(scaler_name, data, cols)
        print(f"{scaler_name}: train={train_scaled.shape}, test={test_scaled.shape}")
        print(train_scaled[cols].head(3).round(3))

    for encoder_name in ["one_hot", "ordinal", "target"]:
        train_encoded, test_encoded, encoder = build_encoded_splits(encoder_name, data)
        print(f"{encoder_name}: train={train_encoded.shape}, test={test_encoded.shape}")
        print(f"encoded columns: {encoder['columns']}")
