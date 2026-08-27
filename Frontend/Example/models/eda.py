"""Generic EDA calculations and plot generation for any uploaded CSV file."""

import base64
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def _chart(title):
    buffer = BytesIO()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    plt.close()
    return title, base64.b64encode(buffer.getvalue()).decode("ascii")


def _make_charts(df):
    """Create useful plots from the columns available in the uploaded dataset."""
    sns.set_theme(style="whitegrid")
    numeric = list(df.select_dtypes(include="number").columns)
    categorical = list(df.select_dtypes(exclude="number").columns)
    charts = []

    # Count plot for the first categorical column (or PlacementStatus when present).
    count_column = (
        "PlacementStatus"
        if "PlacementStatus" in df.columns
        else (categorical[0] if categorical else None)
    )
    if count_column:
        plt.figure(figsize=(6, 4))
        sns.countplot(
            data=df,
            x=count_column,
            order=df[count_column].value_counts().head(12).index,
            color="#3498db",
        )
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Count")
        charts.append(_chart(f"Distribution of {count_column}"))

    # Histogram for the first numeric column (CGPA when it exists).
    histogram_column = (
        "CGPA" if "CGPA" in df.columns else (numeric[0] if numeric else None)
    )
    if histogram_column:
        plt.figure(figsize=(6, 4))
        sns.histplot(data=df, x=histogram_column, bins=20, kde=True, color="#3498db")
        charts.append(_chart(f"Distribution of {histogram_column}"))

    # Pie chart for a categorical column with a manageable number of values.
    pie_column = "Gender" if "Gender" in df.columns else next(
        (column for column in categorical if df[column].nunique() <= 10),
        None,
    )
    if pie_column:
        counts = df[pie_column].value_counts().head(10)
        plt.figure(figsize=(6, 4))
        plt.pie(
            counts,
            labels=counts.index.astype(str),
            autopct="%1.1f%%",
            startangle=90,
        )
        charts.append(_chart(f"Distribution of {pie_column}"))

    # Scatter plot when at least two numeric fields exist.
    scatter_x = "CGPA" if "CGPA" in numeric else (numeric[0] if numeric else None)
    scatter_y = (
        "AttendancePercent"
        if "AttendancePercent" in numeric
        else (numeric[1] if len(numeric) > 1 else None)
    )
    if scatter_x and scatter_y and scatter_x != scatter_y:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(
            data=df,
            x=scatter_x,
            y=scatter_y,
            alpha=0.35,
            s=18,
            color="#2980b9",
        )
        charts.append(_chart(f"{scatter_x} vs {scatter_y}"))

    # Box plot compares a numeric field across categories.
    box_category = count_column
    if box_category and histogram_column and df[box_category].nunique() <= 12:
        plt.figure(figsize=(6, 4))
        sns.boxplot(
            data=df,
            x=box_category,
            y=histogram_column,
            hue=box_category,
            legend=False,
            palette="Set2",
        )
        plt.xticks(rotation=30, ha="right")
        charts.append(_chart(f"{histogram_column} by {box_category}"))

    # Compare the first two categorical columns, if both exist.
    if len(categorical) >= 2:
        first, second = categorical[:2]
        if df[first].nunique() <= 12 and df[second].nunique() <= 12:
            plt.figure(figsize=(6, 4))
            sns.countplot(data=df, x=first, hue=second, palette="Set1")
            plt.xticks(rotation=30, ha="right")
            charts.append(_chart(f"{first} by {second}"))

    # Multivariate view: a correlation heatmap for the most useful numeric fields.
    if len(numeric) >= 2:
        heatmap_columns = numeric[:12]
        correlation = df[heatmap_columns].corr(numeric_only=True)
        plt.figure(figsize=(9, 7))
        sns.heatmap(
            correlation,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.4,
            cbar_kws={"shrink": 0.75},
        )
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(fontsize=8)
        charts.append(_chart("Correlation heatmap (multivariate analysis)"))
    return charts


def build_eda_report(df):
    duplicate_rows = df[df.duplicated(keep=False)]
    missing = df.isnull().sum()
    missing_rows = [
        {
            "column": column,
            "missing": int(count),
            "percentage": round(count / len(df) * 100, 2),
        }
        for column, count in missing.items()
        if count > 0
    ]
    numerical_summary = (
        df.select_dtypes(include="number")
        .describe()
        .T.round(2)
        .reset_index()
        .rename(columns={"index": "column"})
        .to_dict(orient="records")
    )
    numeric_columns = list(df.select_dtypes(include="number").columns)
    categorical_columns = list(df.select_dtypes(exclude="number").columns)

    # IQR rule: values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR are outliers.
    outliers = []
    for column in numeric_columns:
        series = df[column].dropna()
        q1, q3 = series.quantile(.25), series.quantile(.75)
        iqr = q3 - q1
        count = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
        outliers.append(
            {
                "column": column,
                "outliers": count,
                "percentage": round(count / len(series) * 100, 2),
            }
        )
    categorical_summary = [
        {
            "column": column,
            "unique_values": int(df[column].nunique()),
            "most_common": (
                str(df[column].mode().iloc[0])
                if not df[column].mode().empty
                else "-"
            ),
        }
        for column in categorical_columns
    ]
    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "duplicate_count": int(df.duplicated().sum()),
        "missing_cells": int(missing.sum()),
        "missing_rows": missing_rows,
        "duplicate_preview": duplicate_rows.head(10).to_dict(orient="records"),
        "columns": list(df.columns),
        "numerical_summary": numerical_summary,
        "summary_columns": ["column", "count", "mean", "std", "min", "25%", "50%", "75%", "max"],
        "head_rows": df.head(10).to_dict(orient="records"),
        "tail_rows": df.tail(10).to_dict(orient="records"),
        "data_types": [
            {"column": column, "dtype": str(dtype)}
            for column, dtype in df.dtypes.items()
        ],
        "numeric_count": len(numeric_columns),
        "categorical_count": len(categorical_columns),
        "outliers": outliers,
        "categorical_summary": categorical_summary,
        "charts": _make_charts(df),
    }
    