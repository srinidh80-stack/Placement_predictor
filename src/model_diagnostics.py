from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import config
from src.data_utils import load_cleaned


def analyze_bias_variance(df=None, max_depth_range=None, random_state=42):
    """Analyze Bias-Variance Tradeoff across varying Decision Tree maximum depths."""
    data = load_cleaned() if df is None else df.copy()
    
    feature_cols = [
        "CGPA", "CodingTestScore", "MockInterviewScore", "AptitudeTestScore",
        "AttendancePercent", "SoftSkillsRating", "Internships", "Projects"
    ]
    features = [f for f in feature_cols if f in data.columns]
    
    X = data[features].fillna(data[features].median())
    y = data[config.TARGET_CLASSIFICATION].astype(int)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    if max_depth_range is None:
        max_depth_range = list(range(1, 19))
        
    train_errors = []
    val_errors = []
    train_accs = []
    val_accs = []
    
    for depth in max_depth_range:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=random_state)
        dt.fit(X_train, y_train)
        
        tr_acc = accuracy_score(y_train, dt.predict(X_train))
        v_acc = accuracy_score(y_val, dt.predict(X_val))
        
        train_accs.append(round(tr_acc * 100, 2))
        val_accs.append(round(v_acc * 100, 2))
        train_errors.append(round((1.0 - tr_acc) * 100, 2))
        val_errors.append(round((1.0 - v_acc) * 100, 2))
        
    best_idx = int(np.argmin(val_errors))
    optimal_depth = max_depth_range[best_idx]
    best_val_acc = val_accs[best_idx]
    
    summary = {
        "depths": max_depth_range,
        "train_errors": train_errors,
        "val_errors": val_errors,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "optimal_depth": optimal_depth,
        "best_val_acc": best_val_acc,
        "underfitting_depth": 1,
        "overfitting_depth": max_depth_range[-1],
        "underfitting_error": val_errors[0],
        "overfitting_gap": round(train_accs[-1] - val_accs[-1], 2),
    }
    
    # Generate Bias-Variance Curve Plot
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=120)
    ax.plot(max_depth_range, train_errors, marker="o", color="#0f9f8f", linewidth=2.2, label="Training Error (Bias ↓ with depth)")
    ax.plot(max_depth_range, val_errors, marker="s", color="#e05242", linewidth=2.2, linestyle="--", label="Validation Error (Variance ↑ when overfitted)")
    
    ax.axvline(optimal_depth, color="#f5b84b", linestyle=":", linewidth=2, label=f"Optimal Tradeoff (Depth = {optimal_depth})")
    ax.scatter([optimal_depth], [val_errors[best_idx]], color="#f5b84b", s=130, edgecolor="#21312f", zorder=5)
    
    # Annotation regions
    ax.text(2, max(val_errors) * 0.9, "High Bias\n(Underfitting)", color="#21312f", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff2ea", edgecolor="#e05242", alpha=0.8))
    ax.text(max_depth_range[-3], max(val_errors) * 0.9, "High Variance\n(Overfitting)", color="#21312f", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff2ea", edgecolor="#e05242", alpha=0.8))
    
    ax.set_title("Bias-Variance Tradeoff: Decision Tree Depth vs Error Rate", fontsize=11, weight="bold", color="#21312f", pad=12)
    ax.set_xlabel("Tree Maximum Depth (Model Complexity)", fontsize=9.5, weight="bold", color="#21312f")
    ax.set_ylabel("Misclassification Error Rate (%)", fontsize=9.5, weight="bold", color="#21312f")
    ax.legend(frameon=True, loc="center right", fontsize=8.5)
    plt.tight_layout()
    plot_path = config.PLOTS_DIR / "bias_variance_curve.png"
    fig.savefig(plot_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    
    summary["plot_filename"] = "bias_variance_curve.png"
    
    # Save text report
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rep_path = config.REPORTS_DIR / "bias_variance_report.txt"
    with open(rep_path, "w") as f:
        f.write("BIAS-VARIANCE TRADEOFF REPORT (50,000 PLACEMENT RECORDS)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Optimal Tree Depth: {optimal_depth}\n")
        f.write(f"Best Validation Accuracy: {best_val_acc}%\n")
        f.write(f"Underfitting (Depth 1):\n")
        f.write(f"  Train Error: {train_errors[0]}%, Val Error: {val_errors[0]}% (High Bias)\n")
        f.write(f"Overfitting (Depth {max_depth_range[-1]}):\n")
        f.write(f"  Train Error: {train_errors[-1]}%, Val Error: {val_errors[-1]}%\n")
        f.write(f"  Generalization Gap: {summary['overfitting_gap']}% (High Variance)\n")
        f.write(f"Diagnostic Plot: {plot_path.name}\n")
    summary["report_path"] = str(rep_path)
    
    return summary


if __name__ == "__main__":
    print("=" * 70)
    print("  BIAS-VARIANCE TRADEOFF ANALYSIS - PLACEMENT PREDICTION")
    print("=" * 70)
    df = load_cleaned()
    bv = analyze_bias_variance(df)
    print(f"Optimal Depth: {bv['optimal_depth']} (Validation Accuracy: {bv['best_val_acc']}%)")
    print(f"Report saved: {bv['report_path']}")
    print(f"Plot saved: {bv['plot_filename']}")
    print("=" * 70)
