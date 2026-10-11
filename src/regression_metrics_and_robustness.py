"""
Regression Metrics & Outlier Robustness Analysis
Course: Machine Learning (25SC2107E) - Module 5: Evaluation & Validation
Reference: Session: Regression Metrics & Robustness

Evaluates continuous Salary Package prediction for placed students:
  1. Compares Linear Regression and Random Forest Regressor across key metrics:
     - MAE (Mean Absolute Error)
     - RMSE (Root Mean Squared Error)
     - MedAE (Median Absolute Error - non-parametric outlier-robust metric)
     - MAPE (Mean Absolute Percentage Error)
     - R-squared (R2 variance explained)
     - RMSE / MAE ratio (diagnosing variance in error distributions)
  2. Residual Diagnostics:
     - Predicted vs Actual scatter plot
     - Residuals vs Predicted scatter plot
     - Residual histogram & normality distribution
  3. Empirical Outlier Robustness Experiment:
     - Injects synthetic extreme errors (+15 LPA) across contamination levels (0% to 10%)
     - Proves RMSE escalates quadratically while MedAE remains resilient.
"""

from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, mean_absolute_percentage_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

# Ensure output directories exist
config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_placed_regression_data():
    """Filter train and val splits to placed students only where Salary Package is defined."""
    train = pd.read_csv(config.SPLITS_DIR / "train.csv")
    val = pd.read_csv(config.SPLITS_DIR / "val.csv")
    salary_col = "Salary Package"

    train_p = train[train[config.TARGET_COLUMN] == 1].copy()
    val_p = val[val[config.TARGET_COLUMN] == 1].copy()
    feature_cols = [c for c in config.NUMERIC_COLUMNS if c != salary_col] + config.CATEGORY_COLUMNS + config.YES_NO_COLUMNS + ["CollegeTier", "CGPA_Tier"]

    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    ordinal = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(categories=[["Tier3", "Tier2", "Tier1"], ["Low", "Mid", "High"]]))
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    prep = ColumnTransformer([
        ("numeric", numeric, [c for c in config.NUMERIC_COLUMNS if c != salary_col]),
        ("ordinal", ordinal, ["CollegeTier", "CGPA_Tier"]),
        ("categorical", categorical, config.CATEGORY_COLUMNS + config.YES_NO_COLUMNS)
    ])

    return {
        "train_p": train_p, "val_p": val_p,
        "feature_cols": feature_cols,
        "prep": prep,
        "salary_col": salary_col
    }


def train_and_evaluate_regressors(data, random_state=42):
    """Fit regression models and compute comprehensive error metrics."""
    train_p, val_p = data["train_p"], data["val_p"]
    feature_cols, prep = data["feature_cols"], data["prep"]
    salary_col = data["salary_col"]

    models = {
        "Linear Regression": Pipeline([("prep", prep), ("m", LinearRegression())]),
        "Random Forest Regressor": Pipeline([("prep", prep), ("m", RandomForestRegressor(n_estimators=100, max_depth=8, random_state=random_state, n_jobs=1))]),
    }
    preds, metrics_rows = {}, []
    y_val = val_p[salary_col].values

    for name, pipe in models.items():
        pipe.fit(train_p[feature_cols], train_p[salary_col])
        p = pipe.predict(val_p[feature_cols])
        preds[name] = p
        mae = float(mean_absolute_error(y_val, p))
        mse = float(mean_squared_error(y_val, p))
        rmse = float(np.sqrt(mse))
        medae = float(median_absolute_error(y_val, p))
        mape = float(mean_absolute_percentage_error(y_val, p) * 100)
        r2 = float(r2_score(y_val, p))
        metrics_rows.append((name, mae, rmse, medae, mape, r2))

    best_name = min(metrics_rows, key=lambda r: r[1])[0]
    return {"preds": preds, "metrics_rows": metrics_rows, "best_name": best_name, "y_val": y_val}


def test_outlier_robustness(y_val, best_pred, outlier_offset=15.0, contamination_levels=(0.0, 0.01, 0.02, 0.05, 0.10), random_state=42):
    """Inject synthetic large errors into predictions and observe metric degradation rates."""
    rng = np.random.RandomState(random_state)
    robustness_rows = []
    for frac in contamination_levels:
        corrupted = best_pred.copy()
        n_corrupt = int(frac * len(corrupted))
        if n_corrupt:
            idx = rng.choice(len(corrupted), n_corrupt, replace=False)
            corrupted[idx] = corrupted[idx] - outlier_offset
        mae = float(mean_absolute_error(y_val, corrupted))
        rmse = float(np.sqrt(mean_squared_error(y_val, corrupted)))
        medae = float(median_absolute_error(y_val, corrupted))
        robustness_rows.append((frac, mae, rmse, medae))
    return robustness_rows


def generate_regression_diagrams(y_val, best_name, best_pred, robustness_rows, outlier_offset=15.0):
    """Generate diagnostic plots and save to config.PLOTS_DIR."""
    plots = {}
    residuals = y_val - best_pred

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].scatter(best_pred, y_val, alpha=0.3, s=10, color="#4C72B0")
    lims = [min(y_val.min(), best_pred.min()), max(y_val.max(), best_pred.max())]
    axes[0].plot(lims, lims, "r--", linewidth=1.5)
    axes[0].set_xlabel("Predicted Salary Package (LPA)", fontweight="bold")
    axes[0].set_ylabel("Actual Salary Package (LPA)", fontweight="bold")
    axes[0].set_title(f"{best_name}: Predicted vs Actual", fontweight="bold")

    axes[1].scatter(best_pred, residuals, alpha=0.3, s=10, color="#DD8452")
    axes[1].axhline(0, color="red", linewidth=1.5)
    axes[1].set_xlabel("Predicted Salary Package (LPA)", fontweight="bold")
    axes[1].set_ylabel("Residual (Actual - Predicted)", fontweight="bold")
    axes[1].set_title("Residuals vs Predicted", fontweight="bold")

    axes[2].hist(residuals, bins=40, color="#55A868", edgecolor="black", alpha=0.7)
    axes[2].axvline(0, color="red", linewidth=1.5)
    axes[2].set_xlabel("Residual (LPA)", fontweight="bold")
    axes[2].set_title(f"Residual Distribution (MAE {mean_absolute_error(y_val, best_pred):.2f})", fontweight="bold")
    plt.tight_layout()
    p1 = config.PLOTS_DIR / "residual_diagnostics.png"
    plt.savefig(p1, dpi=150)
    plt.close()
    plots["residual_diagnostics"] = p1.name

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fracs = [r[0] * 100 for r in robustness_rows]
    ax.plot(fracs, [r[1] for r in robustness_rows], marker="o", label="MAE", linewidth=1.8)
    ax.plot(fracs, [r[2] for r in robustness_rows], marker="s", label="RMSE", linewidth=1.8)
    ax.plot(fracs, [r[3] for r in robustness_rows], marker="^", label="Median Absolute Error (MedAE)", linewidth=1.8)
    ax.set_xlabel(f"% of Predictions Corrupted by {outlier_offset} LPA Error", fontweight="bold")
    ax.set_ylabel("Metric Value (LPA)", fontweight="bold")
    ax.set_title(f"Outlier Sensitivity Benchmark ({best_name})", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    p2 = config.PLOTS_DIR / "outlier_robustness.png"
    plt.savefig(p2, dpi=150)
    plt.close()
    plots["outlier_robustness"] = p2.name

    return plots


def save_regression_report(metrics_rows, best_name, robustness_rows, n_tr, n_va, outlier_offset):
    """Write comprehensive diagnostic text report to Output/Report/."""
    report_path = config.REPORTS_DIR / "regression_metrics_report.txt"
    base_mae, base_rmse, base_medae = robustness_rows[0][1], robustness_rows[0][2], robustness_rows[0][3]
    worst_mae, worst_rmse, worst_medae = robustness_rows[-1][1], robustness_rows[-1][2], robustness_rows[-1][3]

    with open(report_path, "w") as f:
        f.write("REGRESSION METRICS AND OUTLIER ROBUSTNESS REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Task: Predict continuous Salary Package on placed cohort ({n_tr:,} train, {n_va:,} val).\n\n")
        f.write(f"{'Model':<24}{'MAE':>8}{'RMSE':>8}{'MedAE':>8}{'MAPE%':>8}{'R2':>8}{'RMSE/MAE':>10}\n")
        for name, mae, rmse, medae, mape, r2 in metrics_rows:
            f.write(f"{name:<24}{mae:>8.3f}{rmse:>8.3f}{medae:>8.3f}{mape:>8.1f}{r2:>8.4f}{rmse/mae:>10.3f}\n")

        f.write(f"\nBest Model by MAE: {best_name}\n\n")
        f.write(f"Empirical Outlier Robustness Experiment ({best_name}, {outlier_offset} LPA error injected):\n")
        f.write(f"{'% Corrupted':>12}{'MAE':>8}{'RMSE':>8}{'MedAE':>8}\n")
        for frac, mae, rmse, medae in robustness_rows:
            f.write(f"{int(frac*100):>11}%{mae:>8.3f}{rmse:>8.3f}{medae:>8.3f}\n")

        f.write(f"\nMetric Escalation at 10% Contamination:\n")
        f.write(f"  MAE change:   {((worst_mae/base_mae)-1)*100:+.1f}%\n")
        f.write(f"  RMSE change:  {((worst_rmse/base_rmse)-1)*100:+.1f}%\n")
        f.write(f"  MedAE change: {((worst_medae/base_medae)-1)*100:+.1f}%\n\n")

        f.write("Methodological Takeaways:\n")
        f.write("  1. RMSE heavily penalizes large errors due to squaring; always inspect RMSE/MAE ratio.\n")
        f.write("  2. MedAE remains immune to small percentages of extreme outliers.\n")
        f.write("  3. R-squared reflects proportion of variance explained and serves as standardized scale-free benchmark.\n")
    return str(report_path)


def run_regression_metrics_pipeline(outlier_offset=15.0, random_state=42):
    """End-to-end execution of regression metrics evaluation and robustness analysis."""
    print("=" * 80)
    print("  REGRESSION METRICS & OUTLIER ROBUSTNESS PIPELINE")
    print("  Target: Salary Package (LPA) | Placed Cohort Analysis")
    print("=" * 80)
    start_time = time.perf_counter()

    data = load_placed_regression_data()
    print(f"\n[1/4] Loaded placed cohort: {len(data['train_p']):,} Train | {len(data['val_p']):,} Validation")

    print("\n[2/4] Training Linear Regression & Random Forest models...")
    eval_res = train_and_evaluate_regressors(data, random_state=random_state)
    for m in eval_res["metrics_rows"]:
        print(f"      {m[0]:<24}: MAE={m[1]:.3f} | RMSE={m[2]:.3f} | MedAE={m[3]:.3f} | R2={m[5]:.4f}")

    best_name = eval_res["best_name"]
    best_pred = eval_res["preds"][best_name]
    y_val = eval_res["y_val"]

    print(f"\n[3/4] Testing outlier robustness against {outlier_offset} LPA synthetic error injection...")
    robustness_rows = test_outlier_robustness(y_val, best_pred, outlier_offset=outlier_offset, random_state=random_state)
    for r in robustness_rows:
        print(f"      {int(r[0]*100):>2}% Corrupted: MAE={r[1]:.3f} | RMSE={r[2]:.3f} | MedAE={r[3]:.3f}")

    print("\n[4/4] Generating diagnostic plots and saving report...")
    plots = generate_regression_diagrams(y_val, best_name, best_pred, robustness_rows, outlier_offset=outlier_offset)
    rep_path = save_regression_report(eval_res["metrics_rows"], best_name, robustness_rows, len(data["train_p"]), len(data["val_p"]), outlier_offset)

    elapsed = round(time.perf_counter() - start_time, 2)
    print(f"\n[DONE] Regression metrics pipeline completed in {elapsed}s.")
    print(f"       Plots: {list(plots.values())}")
    print(f"       Report: {Path(rep_path).name}")

    return {
        "eval_res": eval_res,
        "robustness_rows": robustness_rows,
        "plots": plots,
        "report_path": rep_path
    }


if __name__ == "__main__":
    run_regression_metrics_pipeline()
