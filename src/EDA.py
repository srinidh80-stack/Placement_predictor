"""Generate placement EDA statistics and reusable chart files."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config
from src.data_utils import load_raw, numeric_cols

PLOT_FILENAMES = [
    "placement_status.png", "missing_values.png", "cgpa_distribution.png",
    "cgpa_histogram.png", "attendance_histogram.png", "cgpa_boxplot_by_status.png",
    "cgpa_attendance.png", "placement_rate_by_tier.png", "top_correlations.png",
    "sgpa_trend.png", "correlation_heatmap.png",
]


def _data(df):
    return load_raw() if df is None else df


def _save(fig, filename):
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.PLOTS_DIR / filename, bbox_inches="tight", dpi=120)
    plt.close(fig)


def _plot_histogram(data, column, filename, title, bins=20):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=data, x=column, bins=bins, kde=True, ax=ax, color="#3867e8")
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    _save(fig, filename)


def _plot_boxplot(data, value_column, category_column, filename, title):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=data, x=category_column, y=value_column, ax=ax, palette="Set2")
    ax.set_title(title)
    ax.set_xlabel(category_column)
    ax.set_ylabel(value_column)
    ax.tick_params(axis="x", rotation=20)
    _save(fig, filename)


def generate_all_plots(df=None):
    """Create the dashboard charts from the supplied or default dataset."""
    data = _data(df)
    sns.set_theme(style="whitegrid")
    missing = data.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=missing.index, y=missing.values, ax=ax, color="#e76f51")
        ax.set_title("Missing values by column")
        ax.set_ylabel("Missing records")
        ax.tick_params(axis="x", rotation=35)
        _save(fig, "missing_values.png")
    if "PlacementStatus" in data:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.countplot(data=data, x="PlacementStatus", hue="PlacementStatus", legend=False, ax=ax, palette="Set2")
        ax.set_title("Placement status distribution")
        _save(fig, "placement_status.png")
    if "CGPA" in data:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(data=data, x="CGPA", kde=True, ax=ax, color="#3867e8")
        ax.set_title("CGPA distribution")
        _save(fig, "cgpa_distribution.png")
        _plot_histogram(data, "CGPA", "cgpa_histogram.png", "CGPA histogram")
    if "AttendancePercent" in data:
        _plot_histogram(data, "AttendancePercent", "attendance_histogram.png", "Attendance histogram")
    if {"CGPA", "PlacementStatus"}.issubset(data.columns):
        _plot_boxplot(data, "CGPA", "PlacementStatus", "cgpa_boxplot_by_status.png", "CGPA by placement status")
    if {"CGPA", "AttendancePercent"}.issubset(data.columns):
        fig, ax = plt.subplots(figsize=(7, 4))
        hue = "PlacementStatus" if "PlacementStatus" in data else None
        scatter_data = data.sample(n=min(5000, len(data)), random_state=42)
        sns.scatterplot(data=scatter_data, x="CGPA", y="AttendancePercent", hue=hue, alpha=.3, s=14, ax=ax)
        ax.set_title("CGPA and attendance")
        _save(fig, "cgpa_attendance.png")
    if {"CollegeTier", "PlacementStatus"}.issubset(data.columns):
        rates = data.groupby("CollegeTier")["PlacementStatus"].mean().mul(100).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x=rates.index, y=rates.values, ax=ax, color="#16b8a6")
        ax.set_title("Placement rate by college tier")
        ax.set_ylabel("Placement rate (%)")
        _save(fig, "placement_rate_by_tier.png")
    values = numeric_cols(data)
    if values and "PlacementStatus" in data:
        corr = data[values + ["PlacementStatus"]].corr(numeric_only=True)
        top = corr["PlacementStatus"].drop("PlacementStatus").abs().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=top.values, y=top.index, ax=ax, color="#3867e8")
        ax.set_title("Top feature correlations with placement")
        ax.set_xlabel("Absolute correlation")
        _save(fig, "top_correlations.png")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Feature correlation heatmap")
        _save(fig, "correation_heatmap.png")
    sgpas = [f"SGPA_Sem{i}" for i in range(1, 9) if f"SGPA_Sem{i}" in data]
    if sgpas and "PlacementStatus" in data:
        trend = data.groupby("PlacementStatus")[sgpas].mean()
        fig, ax = plt.subplots(figsize=(8, 4))
        for status, values in trend.iterrows():
            ax.plot(range(1, len(sgpas) + 1), values, marker="o", label="Placed" if status == 1 else "Not placed")
        ax.set_title("Average SGPA trend by placement status")
        ax.set_xlabel("Semester")
        ax.set_ylabel("Average SGPA")
        ax.legend()
        _save(fig, "sgpa_trend.png")


def get_overview_stats(df=None):
    data = _data(df)
    missing = data.isna().sum()
    return {
        "rows": len(data), "cols": len(data.columns),
        "duplicate_rows": int(data.duplicated().sum()),
        "duplicate_ids": int(data.duplicated(subset=[config.ID_COL]).sum()) if config.ID_COL in data else 0,
        "duplicate_preview": data[data.duplicated(keep=False)].head(10).to_dict(orient="records"),
        "missing_table": [{"column": c, "count": int(v), "percent": round(v / len(data) * 100, 2)} for c, v in missing[missing > 0].sort_values(ascending=False).items()],
        "placement_counts": data["PlacementStatus"].value_counts().to_dict() if "PlacementStatus" in data else {},
        "placement_pct": data["PlacementStatus"].value_counts(normalize=True).mul(100).round(2).to_dict() if "PlacementStatus" in data else {},
        "anomaly_rate": round(data["IsAnomaly"].mean() * 100, 2) if "IsAnomaly" in data else None,
        "columns": list(data.columns),
        "data_types": [{"column": c, "dtype": str(t)} for c, t in data.dtypes.items()],
        "head_rows": data.head(10).to_dict(orient="records"),
        "tail_rows": data.tail(10).to_dict(orient="records"),
    }


def get_univariate_stats(df=None):
    data = _data(df)
    cols = numeric_cols(data)
    desc = data[cols].describe().T.round(2) if cols else pd.DataFrame()
    return {"numeric_table": [{"column": c, **row.to_dict(), "skew": round(float(data[c].skew()), 2)} for c, row in desc.iterrows()], "categorical_freq": {c: data[c].value_counts(dropna=False).head(12).to_dict() for c in config.CATEGORICAL_COLS if c in data}}


def get_bivariate_stats(df=None):
    data = _data(df)
    groups = [c for c in ["CollegeTier", "Stream", "Specialisation", "Hostel", "CGPA_Tier"] if c in data]
    rates = {c: data.groupby(c)["PlacementStatus"].mean().mul(100).round(2).sort_values(ascending=False).to_dict() for c in groups} if "PlacementStatus" in data else {}
    outliers = {}
    for c in numeric_cols(data):
        q1, q3 = data[c].quantile(.25), data[c].quantile(.75)
        outliers[c] = int(((data[c] < q1 - 1.5 * (q3 - q1)) | (data[c] > q3 + 1.5 * (q3 - q1))).sum())
    return {"placement_rate": rates, "outliers": dict(sorted(outliers.items(), key=lambda item: item[1], reverse=True))}


def get_multivariate_stats(df=None):
    data = _data(df)
    sgpas = [f"SGPA_Sem{i}" for i in range(1, 9) if f"SGPA_Sem{i}" in data]
    return {"salary_desc": data.loc[data["PlacementStatus"] == 1, "Salary Package"].describe().round(2).to_dict() if {"PlacementStatus", "Salary Package"}.issubset(data) else {}, "sgpa_trend": data.groupby("PlacementStatus")[sgpas].mean().round(2).to_dict(orient="index") if sgpas and "PlacementStatus" in data else {}}


def get_correlation_stats(df=None):
    data = _data(df)
    cols = numeric_cols(data)
    if "PlacementStatus" not in data or not cols:
        return {"top_correlations": {}}
    corr = data[cols + ["PlacementStatus"]].corr(numeric_only=True)["PlacementStatus"].drop("PlacementStatus")
    return {"top_correlations": corr.sort_values(ascending=False).round(3).to_dict()}


if __name__ == "__main__":
    dataset = load_raw()
    generate_all_plots(dataset)
    overview = get_overview_stats(dataset)
    print("EDA completed")
    print(f"Dataset shape: {overview['rows']} rows x {overview['cols']} columns")
    print(f"Duplicate rows: {overview['duplicate_rows']}")
    print(f"Generated charts: {config.PLOTS_DIR}")
