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
    "attendance_histogram.png", "cgpa_boxplot_by_status.png",
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
    sns.boxplot(data=data, x=category_column, y=value_column, hue=category_column, legend=False, ax=ax, palette="Set2")
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
        _save(fig, "correlation_heatmap.png")
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


def generate_eda_report(df=None, filepath=None):
    """Generate and save a comprehensive EDA text summary report."""
    data = _data(df)
    out_path = Path(filepath) if filepath else config.EDA_REPORT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    overview = get_overview_stats(data)
    univariate = get_univariate_stats(data)
    bivariate = get_bivariate_stats(data)
    corr_stats = get_correlation_stats(data)
    
    lines = []
    lines.append("=" * 70)
    lines.append("       EXPLORATORY DATA ANALYSIS (EDA) SUMMARY REPORT")
    lines.append("                  Placement Prediction")
    lines.append("=" * 70)
    lines.append("")
    
    lines.append("1. DATASET OVERVIEW")
    lines.append("-" * 50)
    lines.append(f"  * Total Records (Rows): {overview['rows']:,}")
    lines.append(f"  * Total Features (Columns): {overview['cols']}")
    lines.append(f"  * Duplicate Rows: {overview['duplicate_rows']}")
    if overview.get('duplicate_ids'):
        lines.append(f"  * Duplicate Student IDs: {overview['duplicate_ids']}")
    lines.append("")
    
    lines.append("2. TARGET VARIABLE DISTRIBUTION (PlacementStatus)")
    lines.append("-" * 50)
    if overview['placement_counts']:
        for status, count in overview['placement_counts'].items():
            status_label = "Placed (1)" if status == 1 else "Not Placed (0)"
            pct = overview['placement_pct'].get(status, 0)
            lines.append(f"  * {status_label}: {count:,} ({pct:.2f}%)")
    else:
        lines.append("  * PlacementStatus column not found.")
    lines.append("")
    
    lines.append("3. MISSING VALUES SUMMARY")
    lines.append("-" * 50)
    if overview['missing_table']:
        for item in overview['missing_table']:
            lines.append(f"  * {item['column']}: {item['count']:,} missing ({item['percent']}%)")
    else:
        lines.append("  * No missing values found across all columns.")
    lines.append("")
    
    lines.append("4. NUMERICAL FEATURE DESCRIPTIVE STATISTICS")
    lines.append("-" * 50)
    for col_stat in univariate.get('numeric_table', []):
        col_name = col_stat.get('column', '')
        mean_val = col_stat.get('mean', 'N/A')
        std_val = col_stat.get('std', 'N/A')
        min_val = col_stat.get('min', 'N/A')
        median_val = col_stat.get('50%', 'N/A')
        max_val = col_stat.get('max', 'N/A')
        skew_val = col_stat.get('skew', 'N/A')
        lines.append(f"  * {col_name:<22} | Mean: {mean_val:>7} | Std: {std_val:>7} | Min: {min_val:>6} | Median: {median_val:>6} | Max: {max_val:>6} | Skew: {skew_val:>6}")
    lines.append("")
    
    lines.append("5. CATEGORICAL BREAKDOWN & PLACEMENT RATES")
    lines.append("-" * 50)
    for cat_feature, rates in bivariate.get('placement_rate', {}).items():
        lines.append(f"  * {cat_feature}:")
        for cat_val, rate in rates.items():
            lines.append(f"      - {str(cat_val):<18}: {rate:.2f}% placement rate")
    lines.append("")
    
    lines.append("6. TOP CORRELATIONS WITH PLACEMENT STATUS")
    lines.append("-" * 50)
    for feat, corr_val in corr_stats.get('top_correlations', {}).items():
        direction = "Positive" if corr_val > 0 else "Negative"
        lines.append(f"  * {feat:<22}: {corr_val:>+.4f} ({direction})")
    lines.append("")
    
    lines.append("7. OUTLIER ANALYSIS (IQR Method)")
    lines.append("-" * 50)
    for feat, count in bivariate.get('outliers', {}).items():
        if count > 0:
            lines.append(f"  * {feat:<22}: {count:,} outliers detected")
    lines.append("")
    lines.append("=" * 70)
    lines.append("                       END OF REPORT")
    lines.append("=" * 70)
    
    report_text = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    return report_text


if __name__ == "__main__":
    dataset = load_raw()
    generate_all_plots(dataset)
    report_content = generate_eda_report(dataset)
    overview = get_overview_stats(dataset)
    print("EDA completed")
    print(f"Dataset shape: {overview['rows']} rows x {overview['cols']} columns")
    print(f"Duplicate rows: {overview['duplicate_rows']}")
    print(f"Generated charts: {config.PLOTS_DIR}")
    print(f"Generated report: {config.EDA_REPORT_PATH}")

