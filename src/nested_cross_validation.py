"""
Nested Cross-Validation for Unbiased Model Evaluation
Course: Machine Learning (25SC2107E) - Module 5: Evaluation & Validation
Reference: Session: Nested Cross-Validation

Contrasts ordinary non-nested GridSearchCV against Nested Cross-Validation:
  1. Inner CV loop selects optimal hyperparameters across candidate grid.
  2. Outer CV loop estimates genuine generalization performance of the tuning procedure.
  3. External 5,000-row validation reference set acts as ground-truth benchmark.
  4. Quantifies selection optimism across data sample regimes (e.g. n=200, n=800).
  5. Demonstrates hyperparameter instability across outer test folds.
"""

from pathlib import Path
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.data_utils import ensure_ready_splits

# Ensure output directories exist
config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_nested_cv_data():
    """Load train_ready and val_ready splits."""
    ensure_ready_splits()
    rt = pd.read_csv(config.SPLITS_DIR / "train_ready.csv")
    rv = pd.read_csv(config.SPLITS_DIR / "val_ready.csv")
    drop_cols = ["StudentID", "IsAnomaly", "Salary Package", config.TARGET_COLUMN]
    cols = [c for c in rt.columns if c not in drop_cols]
    X_all, y_all = rt[cols].values, rt[config.TARGET_COLUMN].values
    X_val, y_val = rv[cols].values, rv[config.TARGET_COLUMN].values
    return {"X_all": X_all, "y_all": y_all, "X_val": X_val, "y_val": y_val, "cols": cols}


def run_one_nested_trial(estimator, grid, Xs, ys, X_val, y_val, r):
    """Execute one trial comparing non-nested best_score_, nested outer score, and validation reference."""
    inner = StratifiedKFold(5, shuffle=True, random_state=100 + r)
    outer = StratifiedKFold(5, shuffle=True, random_state=r)
    gs = GridSearchCV(estimator, grid, cv=inner, n_jobs=1).fit(Xs, ys)
    non_nested = float(gs.best_score_)
    reference = float(accuracy_score(y_val, gs.predict(X_val)))
    nested = float(cross_validate(GridSearchCV(estimator, grid, cv=inner, n_jobs=1), Xs, ys, cv=outer)["test_score"].mean())
    return non_nested, nested, reference


def generate_nested_diagrams(searches, sample_sizes, results, n_repeats):
    """Generate and export publication-quality diagnostic diagrams."""
    plots = {}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    w = 0.36
    for ax, model_name in zip(axes, searches):
        xs = np.arange(len(sample_sizes))
        non_m = [np.mean(results[(model_name, n)][:, 0] - results[(model_name, n)][:, 2]) for n in sample_sizes]
        nes_m = [np.mean(results[(model_name, n)][:, 1] - results[(model_name, n)][:, 2]) for n in sample_sizes]
        non_s = [np.std(results[(model_name, n)][:, 0] - results[(model_name, n)][:, 2]) / np.sqrt(n_repeats) for n in sample_sizes]
        nes_s = [np.std(results[(model_name, n)][:, 1] - results[(model_name, n)][:, 2]) / np.sqrt(n_repeats) for n in sample_sizes]
        ax.bar(xs - w / 2, non_m, w, yerr=non_s, capsize=4, color="#C0392B", label="Non-nested (best_score_)")
        ax.bar(xs + w / 2, nes_m, w, yerr=nes_s, capsize=4, color="#4C72B0", label="Nested CV")
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"n={n}" for n in sample_sizes])
        ax.set_title(model_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Sample size used for tuning", fontweight="bold")
    axes[0].set_ylabel("Estimate minus outside reference accuracy\n(positive = optimistic)", fontweight="bold")
    axes[0].legend(fontsize=9)
    plt.suptitle("Non-Nested vs Nested CV: Quantifying Selection Optimism", fontweight="bold", fontsize=12)
    plt.tight_layout()
    p1 = config.PLOTS_DIR / "nested_vs_non_nested_error.png"
    plt.savefig(p1, dpi=150)
    plt.close()
    plots["nested_error"] = p1.name

    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, color in zip(searches, ["#DD8452", "#4C72B0"]):
        d = [np.mean(results[(model_name, n)][:, 0] - results[(model_name, n)][:, 1]) for n in sample_sizes]
        se = [np.std(results[(model_name, n)][:, 0] - results[(model_name, n)][:, 1], ddof=1) / np.sqrt(n_repeats) for n in sample_sizes]
        ax.errorbar(sample_sizes, d, yerr=se, marker="o", capsize=4, color=color, label=model_name)
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xscale("log")
    ax.set_xticks(sample_sizes)
    ax.set_xticklabels(sample_sizes)
    ax.set_xlabel("Sample size (log scale)", fontweight="bold")
    ax.set_ylabel("Non-nested minus nested accuracy", fontweight="bold")
    ax.set_title("Selection Optimism Shrinks With More Data & Constrained Grids", fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    p2 = config.PLOTS_DIR / "selection_optimism_vs_sample_size.png"
    plt.savefig(p2, dpi=150)
    plt.close()
    plots["optimism_vs_sample_size"] = p2.name

    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(-3, 17)
    ax.set_ylim(-0.5, 5.6)
    ax.axis("off")
    for i in range(5):
        y0 = 4.6 - i
        ax.text(-0.2, y0 + 0.35, f"Outer fold {i + 1}", ha="right", va="center", fontsize=9, fontweight="bold")
        for j in range(5):
            is_test = (j == i)
            ax.add_patch(Rectangle((j * 3.2, y0), 3.1, 0.7, facecolor="#C0392B" if is_test else "#D6E4F0", edgecolor="black"))
            ax.text(j * 3.2 + 1.55, y0 + 0.35, "OUTER TEST\n(never used to tune)" if is_test else "outer train",
                    ha="center", va="center", fontsize=7, color="white" if is_test else "black")
    ax.text(8, 5.4, "Outer loop: hold out each fold to SCORE the entire 'tune then fit' process", ha="center", fontsize=10, fontweight="bold")
    ax.text(8, -0.35, "Inner loop: runs 5-fold CV on outer training data to select best hyperparameters", ha="center", fontsize=9, style="italic")
    plt.tight_layout()
    p3 = config.PLOTS_DIR / "nested_cv_structure.png"
    plt.savefig(p3, dpi=150)
    plt.close()
    plots["nested_structure"] = p3.name

    return plots


def save_nested_report(summary_rows, chosen_configs):
    """Write comprehensive diagnostic text report to Output/Report/."""
    report_path = config.REPORTS_DIR / "nested_cv_report.txt"
    with open(report_path, "w") as f:
        f.write("NESTED CROSS-VALIDATION VS GRIDSEARCHCV REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Model':<34}{'n':>6}{'Non-nested err':>16}{'Nested err':>12}{'Optimism removed':>20}{'SE':>9}\n")
        for model_name, n, e_non, e_nes, d, se in summary_rows:
            f.write(f"{model_name:<34}{n:>6}{e_non:>+16.4f}{e_nes:>+12.4f}{d:>+20.4f}{se:>9.4f}\n")

        f.write("\nInner Loop Hyperparameter Selection across Outer Folds (Decision Tree, n=800):\n")
        for i, (cfg, sc) in enumerate(chosen_configs, start=1):
            f.write(f"  Fold {i}: chosen max_depth={cfg['max_depth']}, min_samples_leaf={cfg['min_samples_leaf']} -> outer test acc: {sc:.4f}\n")

        f.write("\nMethodological Insights:\n")
        f.write("  1. Non-nested best_score_ suffers from selection bias: the winner's score is inflated.\n")
        f.write("  2. Nested CV evaluates the tuning PROCEDURE honestly without data leakage.\n")
        f.write("  3. Selection optimism is pronounced when training data is scarce and search grid is large.\n")
    return str(report_path)


def run_nested_cv_pipeline(sample_sizes=(200, 800), n_repeats=3, random_state=42):
    """End-to-end execution of nested cross-validation benchmark."""
    print("=" * 80)
    print("  NESTED CROSS-VALIDATION BENCHMARK PIPELINE")
    print("  Target: PlacementStatus | Reference: 5,000 Validation Rows")
    print("=" * 80)
    start_time = time.perf_counter()

    data = load_nested_cv_data()
    X_all, y_all, X_val, y_val = data["X_all"], data["y_all"], data["X_val"], data["y_val"]

    searches = {
        "Decision tree (grid)": (
            DecisionTreeClassifier(random_state=random_state),
            {"max_depth": [2, 4, 6, 8, None], "min_samples_leaf": [1, 5, 20]}
        ),
        "Logistic regression (tune C)": (
            Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, random_state=random_state))]),
            {"lr__C": np.logspace(-3, 1, 8)}
        ),
    }

    results = {}
    print("\n[1/3] Running Nested vs Non-Nested loops across sample sizes...")
    for model_name, (est, grid) in searches.items():
        for n in sample_sizes:
            rows = []
            for r in range(n_repeats):
                idx = np.random.RandomState(7000 + r).choice(len(X_all), n, replace=False)
                rows.append(run_one_nested_trial(est, grid, X_all[idx], y_all[idx], X_val, y_val, r))
            arr = np.array(rows)
            results[(model_name, n)] = arr
            nn, ne, ref = arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 2].mean()
            print(f"      {model_name:<28} n={n:>4}: Non-nested={nn:.4f} | Nested={ne:.4f} | Ref={ref:.4f} | Optimism={nn-ne:+.4f}")

    summary_rows = []
    for (model_name, n), arr in results.items():
        e_non = float(np.mean(arr[:, 0] - arr[:, 2]))
        e_nes = float(np.mean(arr[:, 1] - arr[:, 2]))
        d = arr[:, 0] - arr[:, 1]
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
        summary_rows.append((model_name, n, e_non, e_nes, float(d.mean()), se))

    print("\n[2/3] Analyzing hyperparameter stability across outer folds (Tree, n=800)...")
    est_tree, grid_tree = searches["Decision tree (grid)"]
    idx_800 = np.random.RandomState(7000).choice(len(X_all), 800, replace=False)
    cv_res = cross_validate(
        GridSearchCV(est_tree, grid_tree, cv=StratifiedKFold(5, shuffle=True, random_state=100), n_jobs=1),
        X_all[idx_800], y_all[idx_800], cv=StratifiedKFold(5, shuffle=True, random_state=0),
        return_estimator=True
    )
    chosen_configs = [(e.best_params_, float(sc)) for e, sc in zip(cv_res["estimator"], cv_res["test_score"])]
    for i, (cfg, sc) in enumerate(chosen_configs, start=1):
        print(f"      Outer Fold {i}: Chosen {cfg} -> Test Acc: {sc:.4f}")

    print("\n[3/3] Generating diagnostic plots and saving report...")
    plots = generate_nested_diagrams(searches, sample_sizes, results, n_repeats)
    rep_path = save_nested_report(summary_rows, chosen_configs)

    elapsed = round(time.perf_counter() - start_time, 2)
    print(f"\n[DONE] Nested CV pipeline completed in {elapsed}s.")
    print(f"       Plots: {list(plots.values())}")
    print(f"       Report: {Path(rep_path).name}")

    return {
        "results": results,
        "summary_rows": summary_rows,
        "chosen_configs": chosen_configs,
        "plots": plots,
        "report_path": rep_path
    }


if __name__ == "__main__":
    run_nested_cv_pipeline()
