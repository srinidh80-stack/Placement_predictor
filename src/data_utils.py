from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import config


def load_raw(dataset_path=None):
    """Load raw dataset from dataset_path or fallback to default configured in config.RAW_DATA_PATH."""
    return pd.read_csv(dataset_path or config.RAW_DATA_PATH)


def detect_iqr_outliers(df, cols=None):
    """Detect and summarize outliers using the 1.5 * IQR method."""
    if cols is None:
        cols = [c for c in ["CGPA", "AttendancePercent", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore"] if c in df.columns]
    
    report = {}
    for col in cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = series[(series < lower) | (series > upper)]
        report[col] = {
            "q1": round(float(q1), 2),
            "q3": round(float(q3), 2),
            "iqr": round(float(iqr), 2),
            "lower_bound": round(float(lower), 2),
            "upper_bound": round(float(upper), 2),
            "outlier_count": int(len(outliers)),
            "outlier_percent": round(float(len(outliers) / len(series) * 100), 2) if len(series) > 0 else 0.0,
        }
    return report


def clean_data(df=None, save=True):
    """Clean data: median imputation, duplicate removal, range clipping, and text normalization."""
    data = load_raw() if df is None else df.copy()
    
    # 1. Median imputation for numeric scores with missing values
    impute_cols = ["Workshops", "AptitudeTestScore", "SoftSkillsRating", "CodingTestScore", "MockInterviewScore"]
    for column in impute_cols:
        if column in data and data[column].isna().any():
            data[column] = data[column].fillna(data[column].median())
            
    # 2. Score bounds clipping (0 to 100 for percentages and test scores)
    score_cols = [c for c in ["AptitudeTestScore", "CodingTestScore", "MockInterviewScore", "AttendancePercent"] if c in data.columns]
    for col in score_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce").clip(0, 100)
        
    # 3. Clean and standardize text columns
    text_cols = [c for c in config.TEXT_COLUMNS if c in data.columns]
    for col in text_cols:
        if data[col].dtype == object or str(data[col].dtype) == "string":
            data[col] = data[col].astype(str).str.strip().str.title()
            
    # 4. Deduplication
    before = len(data)
    data = data.drop_duplicates(subset=[config.ID_COL] if config.ID_COL in data else None)
    
    if save:
        config.CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(config.CLEANED_DATA_PATH, index=False)
        
    return data, before - len(data)


def load_cleaned():
    """Load pre-cleaned dataset or run cleaning if not present."""
    return pd.read_csv(config.CLEANED_DATA_PATH) if config.CLEANED_DATA_PATH.exists() else clean_data()[0]


def calculate_baseline_metrics(df=None):
    """Compute Zero-Rule Baseline metrics for classification (majority class) and regression (mean)."""
    data = load_cleaned() if df is None else df.copy()
    results = {}
    
    # Classification Baseline: Zero-Rule Majority Class
    target_cls = config.TARGET_CLASSIFICATION
    if target_cls in data.columns:
        valid_y = data[target_cls].dropna().astype(int)
        majority_class = int(valid_y.mode()[0])
        baseline_acc = float((valid_y == majority_class).mean())
        results["classification"] = {
            "target": target_cls,
            "majority_class": majority_class,
            "label": "Placed (1)" if majority_class == 1 else "Not Placed (0)",
            "baseline_accuracy": round(baseline_acc * 100, 2),
            "baseline_error": round((1.0 - baseline_acc) * 100, 2),
            "description": "Always guessing the most frequent class provides the minimum accuracy benchmark any ML model must surpass."
        }
        
    # Regression Baseline: Mean predictor
    target_reg = config.TARGET_REGRESSION
    if target_reg in data.columns:
        placed = data[data[target_cls] == 1] if target_cls in data.columns else data
        valid_sal = pd.to_numeric(placed[target_reg], errors="coerce").dropna()
        if len(valid_sal) > 0:
            mean_salary = float(valid_sal.mean())
            mae_baseline = float(np.abs(valid_sal - mean_salary).mean())
            rmse_baseline = float(np.sqrt(np.mean((valid_sal - mean_salary) ** 2)))
            results["regression"] = {
                "target": target_reg,
                "mean_salary": round(mean_salary, 2),
                "baseline_mae": round(mae_baseline, 2),
                "baseline_rmse": round(rmse_baseline, 2),
                "description": "Always predicting the dataset mean salary serves as the baseline error benchmark for regression models."
            }
            
    return results


def create_stratified_splits(df=None, train_size=0.7, val_size=0.1, test_size=0.2, random_state=42):
    """Generate and return 70/10/20 train/validation/test splits."""
    data = load_cleaned() if df is None else df.copy()
    target = config.TARGET_CLASSIFICATION
    strat = data[target] if target in data.columns else None
    
    # First split: train+val (80%) and test (20%)
    train_val, test = train_test_split(
        data, test_size=test_size, stratify=strat, random_state=random_state
    )
    
    # Second split: train (70% of total) and val (10% of total) -> 0.1 / 0.8 = 0.125
    strat_tv = train_val[target] if target in train_val.columns else None
    train, val = train_test_split(
        train_val, test_size=(val_size / (train_size + val_size)), stratify=strat_tv, random_state=random_state
    )
    
    return train, val, test


def generate_and_save_splits(df=None, train_size=0.7, val_size=0.1, test_size=0.2, random_state=42):
    """Generate 70/10/20 train/validation/test splits, save to CSVs, and write split_report.txt."""
    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    train, val, test = create_stratified_splits(df, train_size, val_size, test_size, random_state)
    
    train_path = config.SPLITS_DIR / "train.csv"
    val_path = config.SPLITS_DIR / "val.csv"
    test_path = config.SPLITS_DIR / "test.csv"
    
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)
    
    # Save Split Report
    split_report_path = config.REPORTS_DIR / "split_report.txt"
    with open(split_report_path, "w") as f:
        f.write("TRAIN / VALIDATION / TEST SPLIT REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total samples: {len(train) + len(val) + len(test)}\n")
        f.write(f"Train split: {len(train)} ({train_size*100:.1f}%) -> {train_path.name}\n")
        f.write(f"Validation split: {len(val)} ({val_size*100:.1f}%) -> {val_path.name}\n")
        f.write(f"Test split: {len(test)} ({test_size*100:.1f}%) -> {test_path.name}\n")
        f.write(f"Stratified target: {config.TARGET_CLASSIFICATION}\n")
        f.write(f"Train Placed rate: {train[config.TARGET_CLASSIFICATION].mean()*100:.2f}%\n")
        f.write(f"Val Placed rate: {val[config.TARGET_CLASSIFICATION].mean()*100:.2f}%\n")
        f.write(f"Test Placed rate: {test[config.TARGET_CLASSIFICATION].mean()*100:.2f}%\n")
        
    return {
        "train_len": len(train),
        "val_len": len(val),
        "test_len": len(test),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "test_path": str(test_path),
        "report_path": str(split_report_path)
    }


def save_cleaning_report(df_raw=None, df_clean=None, rows_dropped=0):
    """Write comprehensive cleaning report to Output/Report/cleaning_report.txt."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if df_raw is None:
        df_raw = load_raw()
    if df_clean is None:
        df_clean = load_cleaned()
        
    outliers = detect_iqr_outliers(df_clean)
    report_path = config.REPORTS_DIR / "cleaning_report.txt"
    
    with open(report_path, "w") as f:
        f.write("DATA CLEANING AND OUTLIER REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Rows before cleaning: {len(df_raw)}\n")
        f.write(f"Rows after cleaning:  {len(df_clean)}\n")
        f.write(f"Duplicates removed:   {rows_dropped}\n\n")
        f.write("IQR Outlier Detection (1.5 * IQR bounds):\n")
        for col, stats in outliers.items():
            f.write(f"  - {col:<22}: {stats['outlier_count']} outliers ({stats['outlier_percent']}%) "
                    f"[Bounds: {stats['lower_bound']} to {stats['upper_bound']}]\n")
        f.write("\nValue Clipping Applied:\n")
        f.write("  - Scores and percentages clipped to valid range [0, 100]\n")
        f.write("  - Categorical text columns whitespace-stripped and title-cased\n")
        
    return str(report_path)


def save_baseline_report(df=None):
    """Write baseline zero-rule benchmark report to Output/Report/baseline_report.txt."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = calculate_baseline_metrics(df)
    report_path = config.REPORTS_DIR / "baseline_report.txt"
    
    with open(report_path, "w") as f:
        f.write("ZERO-RULE BASELINE MODEL REPORT\n")
        f.write("=" * 50 + "\n")
        if "classification" in metrics:
            c = metrics["classification"]
            f.write("Classification Baseline (Majority Class):\n")
            f.write(f"  Target column: {c['target']}\n")
            f.write(f"  Majority class: {c['majority_class']} ({c['label']})\n")
            f.write(f"  Baseline Accuracy: {c['baseline_accuracy']}%\n")
            f.write(f"  Baseline Error: {c['baseline_error']}%\n")
            f.write(f"  Rule: Guessing '{c['label']}' every time gives {c['baseline_accuracy']}% accuracy.\n\n")
        if "regression" in metrics:
            r = metrics["regression"]
            f.write("Regression Baseline (Mean Predictor):\n")
            f.write(f"  Target column: {r['target']}\n")
            f.write(f"  Mean Salary Package: {r['mean_salary']} LPA\n")
            f.write(f"  Baseline MAE:  {r['baseline_mae']} LPA\n")
            f.write(f"  Baseline RMSE: {r['baseline_rmse']} LPA\n")
            
    return str(report_path)


def numeric_cols(df):
    excluded = set(config.CATEGORICAL_COLS + config.TARGET_COLS + [config.ID_COL])
    return [c for c in df.select_dtypes(include="number").columns if c not in excluded]


if __name__ == "__main__":
    print("Running data utilities...")
    df_raw = load_raw()
    df_clean, dropped = clean_data(df_raw)
    rep_clean = save_cleaning_report(df_raw, df_clean, dropped)
    print(f"Cleaning report saved: {rep_clean}")
    rep_base = save_baseline_report(df_clean)
    print(f"Baseline report saved: {rep_base}")
    splits = generate_and_save_splits(df_clean)
    print(f"Stratified splits created: Train={splits['train_len']}, Val={splits['val_len']}, Test={splits['test_len']}")
    print(f"Split report saved: {splits['report_path']}")

