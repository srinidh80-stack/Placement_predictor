"""
Grid Search vs Random Search Hyperparameter Tuning
Course: Machine Learning (25SC2107E) - Module 5: Evaluation & Validation
Reference: Session: Grid Search vs Random Search

Empirical validation of Bergstra & Bengio (2012) principles:
  1. Experiment A (Low Dimensionality, 2 Hyperparameters - Decision Tree):
     - Exhaustive GridSearchCV vs budget-matched RandomizedSearchCV.
     - Confirms grid search works well when dimensions are low and resolution is dense.
  2. Experiment B (Higher Dimensionality, 5 Hyperparameters - LightGBM):
     - Demonstrates the curse of dimensionality for grid searches (3^5 = 243 combinations).
     - At equal computational budgets (30 fits), random search explores far more distinct parameter levels.
  3. Visualizations:
     - Experiment A accuracy comparison
     - Experiment B accuracy and parameter coverage comparison
  4. Exports plots to config.PLOTS_DIR and report to config.REPORTS_DIR.
"""

from pathlib import Path
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import loguniform, randint
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, ParameterGrid, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.data_utils import ensure_ready_splits

warnings.filterwarnings("ignore")

# Ensure output directories exist
config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_search_subsample(sample_size=3000, random_state=42):
    """Load train_ready split and draw subsample for tractable CPU evaluation."""
    ensure_ready_splits()
    rt = pd.read_csv(config.SPLITS_DIR / "train_ready.csv")
    cols = [c for c in rt.columns if c not in ["StudentID", "IsAnomaly", "Salary Package", config.TARGET_COLUMN]]
    idx = np.random.RandomState(random_state).choice(len(rt), min(sample_size, len(rt)), replace=False)
    X = rt[cols].iloc[idx].reset_index(drop=True)
    y = rt[config.TARGET_COLUMN].values[idx]
    return {"X": X, "y": y, "cols": cols, "n_samples": len(X)}


def run_experiment_a(X, y, random_state=42):
    """Experiment A: Decision Tree tuning with 2 hyperparameters (Grid vs Random)."""
    cv = StratifiedKFold(5, shuffle=True, random_state=random_state)
    grid_a = {"max_depth": [2, 3, 4, 6, 8, 10], "min_samples_leaf": [1, 5, 10, 20, 40]}
    n_grid_a = len(grid_a["max_depth"]) * len(grid_a["min_samples_leaf"])
    dist_a = {"max_depth": randint(2, 16), "min_samples_leaf": randint(1, 50)}

    t0 = time.time()
    gs_a = GridSearchCV(DecisionTreeClassifier(random_state=random_state), grid_a, cv=cv, n_jobs=1).fit(X, y)
    t_grid_a = time.time() - t0

    t0 = time.time()
    rs_a = RandomizedSearchCV(DecisionTreeClassifier(random_state=random_state), dist_a, n_iter=n_grid_a, cv=cv,
                              random_state=random_state, n_jobs=1).fit(X, y)
    t_rand_a = time.time() - t0

    return {
        "gs_a": gs_a, "rs_a": rs_a,
        "t_grid_a": t_grid_a, "t_rand_a": t_rand_a,
        "n_grid_a": n_grid_a
    }


def run_experiment_b(X, y, budget=25, random_state=42):
    """Experiment B: LightGBM tuning with 5 hyperparameters at equal trial budget."""
    cv = StratifiedKFold(5, shuffle=True, random_state=random_state)
    grid_b = {
        "n_estimators": [50, 150, 300],
        "learning_rate": [0.01, 0.05, 0.2],
        "max_depth": [3, 5, 8],
        "num_leaves": [15, 31, 63],
        "min_child_samples": [5, 20, 60]
    }
    dist_b = {
        "n_estimators": randint(30, 350),
        "learning_rate": loguniform(0.005, 0.3),
        "max_depth": randint(2, 10),
        "num_leaves": randint(7, 80),
        "min_child_samples": randint(3, 70)
    }

    t0 = time.time()
    rs_b = RandomizedSearchCV(LGBMClassifier(random_state=random_state, verbose=-1), dist_b, n_iter=budget, cv=cv,
                              random_state=random_state, n_jobs=1).fit(X, y)
    t_rand_b = time.time() - t0

    all_combos = list(ParameterGrid(grid_b))
    t0 = time.time()
    grid_b_scores = []
    for params in all_combos[:budget]:
        sc = cross_val_score(LGBMClassifier(random_state=random_state, verbose=-1, **params), X, y, cv=cv).mean()
        grid_b_scores.append((params, float(sc)))
    t_grid_b = time.time() - t0

    best_grid_b = max(grid_b_scores, key=lambda r: r[1])
    grid_b_tried = {p: set(c[p] for c, _ in grid_b_scores) for p in grid_b}
    rand_b_tried = {p: len(set(d[p] for d in rs_b.cv_results_["params"])) for p in grid_b}

    return {
        "rs_b": rs_b, "t_rand_b": t_rand_b,
        "grid_b_scores": grid_b_scores, "t_grid_b": t_grid_b,
        "best_grid_b": best_grid_b,
        "grid_b_tried": grid_b_tried, "rand_b_tried": rand_b_tried,
        "budget": budget, "grid_b": grid_b
    }


def generate_search_diagrams(exp_a, exp_b):
    """Generate diagnostic comparison plots and save to config.PLOTS_DIR."""
    plots = {}
    gs_a, rs_a = exp_a["gs_a"], exp_a["rs_a"]
    n_grid_a = exp_a["n_grid_a"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(["Grid search", "Random search"], [gs_a.best_score_, rs_a.best_score_], color=["#C0392B", "#4C72B0"], width=0.5)
    for i, v in enumerate([gs_a.best_score_, rs_a.best_score_]):
        ax.text(i, v + 0.0005, f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(min(gs_a.best_score_, rs_a.best_score_) - 0.003, max(gs_a.best_score_, rs_a.best_score_) + 0.004)
    ax.set_ylabel("Best 5-Fold CV Accuracy", fontweight="bold")
    ax.set_title(f"Experiment A (2 Parameters, {n_grid_a} Fits Each): Equal Budget", fontweight="bold")
    plt.tight_layout()
    p1 = config.PLOTS_DIR / "experiment_a_two_params.png"
    plt.savefig(p1, dpi=150)
    plt.close()
    plots["experiment_a"] = p1.name

    best_grid_b = exp_b["best_grid_b"]
    rs_b = exp_b["rs_b"]
    grid_b_tried = exp_b["grid_b_tried"]
    rand_b_tried = exp_b["rand_b_tried"]
    budget = exp_b["budget"]
    grid_b = exp_b["grid_b"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(["Grid\n(first subset)", f"Random\n({budget} draws)"], [best_grid_b[1], rs_b.best_score_],
                color=["#C0392B", "#4C72B0"], width=0.5)
    for i, v in enumerate([best_grid_b[1], rs_b.best_score_]):
        axes[0].text(i, v + 0.0007, f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")
    axes[0].set_ylim(min(best_grid_b[1], rs_b.best_score_) - 0.006, max(best_grid_b[1], rs_b.best_score_) + 0.006)
    axes[0].set_ylabel("Best 5-Fold CV Accuracy", fontweight="bold")
    axes[0].set_title(f"Experiment B (5 Parameters, Budget={budget})", fontweight="bold")

    pnames = list(grid_b)
    x = np.arange(len(pnames))
    w = 0.35
    axes[1].bar(x - w / 2, [len(grid_b_tried[p]) for p in pnames], w, label=f"Grid ({budget} slice)", color="#C0392B")
    axes[1].bar(x + w / 2, [rand_b_tried[p] for p in pnames], w, label=f"Random ({budget} draws)", color="#4C72B0")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(pnames, rotation=25, ha="right", fontsize=8)
    axes[1].set_ylabel("Distinct Values Sampled", fontweight="bold")
    axes[1].set_title("Hyperparameter Domain Coverage", fontweight="bold")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    p2 = config.PLOTS_DIR / "experiment_b_five_params.png"
    plt.savefig(p2, dpi=150)
    plt.close()
    plots["experiment_b"] = p2.name

    return plots


def save_search_report(exp_a, exp_b, n_samples):
    """Write comprehensive diagnostic text report to Output/Report/."""
    report_path = config.REPORTS_DIR / "search_comparison_report.txt"
    gs_a, rs_a = exp_a["gs_a"], exp_a["rs_a"]
    t_grid_a, t_rand_a = exp_a["t_grid_a"], exp_a["t_rand_a"]
    n_grid_a = exp_a["n_grid_a"]

    best_grid_b = exp_b["best_grid_b"]
    rs_b = exp_b["rs_b"]
    t_grid_b, t_rand_b = exp_b["t_grid_b"], exp_b["t_rand_b"]
    grid_b_tried = exp_b["grid_b_tried"]
    rand_b_tried = exp_b["rand_b_tried"]
    budget = exp_b["budget"]
    grid_b = exp_b["grid_b"]

    with open(report_path, "w") as f:
        f.write("GRID SEARCH VS RANDOM SEARCH BENCHMARK REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Evaluated on {n_samples:,}-row training subsample with 5-fold Stratified CV.\n\n")

        f.write(f"Experiment A - Decision Tree (2 Hyperparameters, {n_grid_a} fits each):\n")
        f.write(f"  Grid Search:   Best CV={gs_a.best_score_:.4f} ({t_grid_a:.1f}s) | Params={gs_a.best_params_}\n")
        f.write(f"  Random Search: Best CV={rs_a.best_score_:.4f} ({t_rand_a:.1f}s) | Params={rs_a.best_params_}\n\n")

        f.write(f"Experiment B - LightGBM (5 Hyperparameters, Matched Budget={budget} fits):\n")
        f.write(f"  Grid Slice:    Best CV={best_grid_b[1]:.4f} ({t_grid_b:.1f}s) | Params={best_grid_b[0]}\n")
        f.write(f"  Random Search: Best CV={rs_b.best_score_:.4f} ({t_rand_b:.1f}s) | Params={rs_b.best_params_}\n\n")

        f.write("Distinct Parameter Values Explored at Equal Budget:\n")
        for p in grid_b:
            f.write(f"  {p:<20} Grid={len(grid_b_tried[p])}  |  Random={rand_b_tried[p]}\n")

        f.write("\nKey Takeaways:\n")
        f.write("  1. In low dimensions, Grid Search easily achieves full factorial coverage.\n")
        f.write("  2. In high dimensions, Grid Search spends budget varying non-influential parameters.\n")
        f.write("  3. Random Search explores the continuous marginal distribution of every parameter independently.\n")
    return str(report_path)


def run_search_comparison_pipeline(sample_size=3000, budget_b=25, random_state=42):
    """End-to-end execution of grid vs random search benchmark."""
    print("=" * 80)
    print("  GRID SEARCH VS RANDOM SEARCH BENCHMARK PIPELINE")
    print("  Models: Decision Tree (2D) & LightGBM (5D)")
    print("=" * 80)
    start_time = time.perf_counter()

    data = load_search_subsample(sample_size=sample_size, random_state=random_state)
    X, y = data["X"], data["y"]
    print(f"\n[1/4] Loaded {data['n_samples']:,} student records for tuning.")

    print("\n[2/4] Running Experiment A (Decision Tree, 2 hyperparameters)...")
    exp_a = run_experiment_a(X, y, random_state=random_state)
    print(f"      Grid Search Best CV:   {exp_a['gs_a'].best_score_:.4f} ({exp_a['t_grid_a']:.1f}s)")
    print(f"      Random Search Best CV: {exp_a['rs_a'].best_score_:.4f} ({exp_a['t_rand_a']:.1f}s)")

    print(f"\n[3/4] Running Experiment B (LightGBM, 5 hyperparameters, budget={budget_b})...")
    exp_b = run_experiment_b(X, y, budget=budget_b, random_state=random_state)
    print(f"      Grid Subset Best CV:   {exp_b['best_grid_b'][1]:.4f} ({exp_b['t_grid_b']:.1f}s)")
    print(f"      Random Search Best CV: {exp_b['rs_b'].best_score_:.4f} ({exp_b['t_rand_b']:.1f}s)")

    print("\n[4/4] Generating comparative plots and saving report...")
    plots = generate_search_diagrams(exp_a, exp_b)
    rep_path = save_search_report(exp_a, exp_b, data["n_samples"])

    elapsed = round(time.perf_counter() - start_time, 2)
    print(f"\n[DONE] Search comparison pipeline completed in {elapsed}s.")
    print(f"       Plots: {list(plots.values())}")
    print(f"       Report: {Path(rep_path).name}")

    return {
        "exp_a": exp_a,
        "exp_b": exp_b,
        "plots": plots,
        "report_path": rep_path
    }


if __name__ == "__main__":
    run_search_comparison_pipeline()
