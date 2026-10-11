"""Bayesian Optimization vs Random Search.

Compares Optuna's Tree-structured Parzen Estimator (TPE) against Random Search
on LightGBM across 5 hyperparameters with identical 30-trial budget and 5-fold CV.
"""

import os
import sys
import time
from pathlib import Path
import warnings

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
from scipy.stats import loguniform, randint
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from lightgbm import LGBMClassifier

import config
from src.data_utils import ensure_ready_splits

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_bayesian_optimization():
    print("=" * 80)
    print("  BAYESIAN OPTIMIZATION VS RANDOM SEARCH PIPELINE")
    print("  Model: LightGBM (5 Hyperparameters) | Budget: 30 Trials")
    print("=" * 80)

    ensure_ready_splits()
    RS = config.RANDOM_STATE
    TARGET = config.TARGET_COLUMN
    BUDGET = 30

    train_ready_path = config.SPLITS_DIR / "train_ready.csv"
    rt = pd.read_csv(train_ready_path)
    cols = [c for c in rt.columns if c not in ["StudentID", "IsAnomaly", "Salary Package", TARGET]]
    
    # 4,000-row subsample for balanced benchmarking speed & statistical power
    sample_size = min(4000, len(rt))
    idx = np.random.RandomState(RS).choice(len(rt), sample_size, replace=False)
    X, y = rt[cols].iloc[idx].reset_index(drop=True), rt[TARGET].values[idx]
    cv = StratifiedKFold(5, shuffle=True, random_state=RS)

    print(f"\n[1/4] Loaded {len(X):,} samples for hyperparameter optimization (Budget={BUDGET} trials).")

    # ---------- Random search ----------
    print("\n[2/4] Running Random Search benchmark (30 trials)...")
    dist = {
        "n_estimators": randint(30, 400),
        "learning_rate": loguniform(0.005, 0.3),
        "max_depth": randint(2, 10),
        "num_leaves": randint(7, 90),
        "min_child_samples": randint(3, 80)
    }
    t0 = time.time()
    rand_search = RandomizedSearchCV(
        LGBMClassifier(random_state=RS, verbose=-1), dist, n_iter=BUDGET, cv=cv,
        random_state=RS, n_jobs=-1, return_train_score=False
    ).fit(X, y)
    t_rand = time.time() - t0
    rand_running_best = np.maximum.accumulate(rand_search.cv_results_["mean_test_score"])
    print(f"      Random Search: Best CV={rand_search.best_score_:.4f} ({t_rand:.1f}s)")
    print(f"      Best params: {rand_search.best_params_}")

    # ---------- Bayesian optimization (Optuna, TPE sampler) ----------
    print("\n[3/4] Running Bayesian Optimization via Optuna TPE (30 trials)...")
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 30, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "num_leaves": trial.suggest_int("num_leaves", 7, 90),
            "min_child_samples": trial.suggest_int("min_child_samples", 3, 80),
        }
        model = LGBMClassifier(random_state=RS, verbose=-1, **params)
        return cross_val_score(model, X, y, cv=cv, n_jobs=-1).mean()

    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RS))
    study.optimize(objective, n_trials=BUDGET, show_progress_bar=False)
    t_bayes = time.time() - t0
    bayes_scores = np.array([t.value for t in study.trials])
    bayes_running_best = np.maximum.accumulate(bayes_scores)
    print(f"      Bayesian Opt:  Best CV={study.best_value:.4f} ({t_bayes:.1f}s)")
    print(f"      Best params: {study.best_params}")

    bayes_won = study.best_value > rand_search.best_score_
    diff = study.best_value - rand_search.best_score_
    print(f"\n      Comparison: Difference={diff:+.4f} (Bayesian {'won' if bayes_won else 'tied/trailed'})")

    target_scores = [0.915, 0.918, 0.920]
    speed_reach = {}
    for target in target_scores:
        rand_hit = int(np.argmax(rand_running_best >= target) + 1) if (rand_running_best >= target).any() else None
        bayes_hit = int(np.argmax(bayes_running_best >= target) + 1) if (bayes_running_best >= target).any() else None
        speed_reach[target] = (rand_hit, bayes_hit)
        print(f"      CV >= {target:.3f}: Random trial={rand_hit or 'never'} vs Bayes trial={bayes_hit or 'never'}")

    # ---------- Plots ----------
    print("\n[4/4] Generating comparative convergence plots...")
    trial_nums = np.arange(1, BUDGET + 1)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(trial_nums, rand_running_best, "o-", color="#C0392B", label=f"Random search (best {rand_search.best_score_:.4f})")
    ax.plot(trial_nums, bayes_running_best, "o-", color="#4C72B0", label=f"Bayesian optimization (best {study.best_value:.4f})")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Best CV accuracy found so far")
    ax.set_title(f"Running Best Score: Random Search vs Bayesian Optimization\n(same {BUDGET}-trial budget, same search space, same CV folds)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plot_running_path = config.PLOTS_DIR / "running_best_comparison.png"
    plt.savefig(plot_running_path, dpi=130)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(trial_nums, [rand_search.cv_results_["mean_test_score"][i] for i in range(BUDGET)],
               color="#C0392B", alpha=0.6, label="Random search trials")
    ax.scatter(trial_nums, bayes_scores, color="#4C72B0", alpha=0.6, label="Bayesian optimization trials")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("CV accuracy of that trial")
    ax.set_title("Per-Trial Scores: Raw Draws for Both Methods (30-trial budget)")
    ax.legend()
    plt.tight_layout()
    plot_trials_path = config.PLOTS_DIR / "per_trial_scores.png"
    plt.savefig(plot_trials_path, dpi=130)
    plt.close()

    # Report
    report_content = []
    report_content.append(f"Bayesian optimization (Optuna/TPE) vs random search, same {len(X):,}-row subsample, same 5-fold CV,")
    report_content.append(f"same {BUDGET}-trial budget, same LightGBM search space.\n")
    report_content.append(f"Random search:          best CV {rand_search.best_score_:.4f}  ({t_rand:.1f}s)  {rand_search.best_params_}")
    report_content.append(f"Bayesian optimization:  best CV {study.best_value:.4f}  ({t_bayes:.1f}s)  {study.best_params}")
    report_content.append(f"Difference: {study.best_value - rand_search.best_score_:+.4f}\n")
    report_content.append("Trials needed to first reach a given CV score:")
    for target in target_scores:
        r_hit, b_hit = speed_reach[target]
        report_content.append(f"  CV >= {target:.3f}: random {r_hit or 'never'}   Bayesian {b_hit or 'never'}")
    report_content.append("\nKey Findings:")
    if bayes_won:
        report_content.append("  - Bayesian optimization reached higher peak accuracy and identified higher-performing configurations faster.")
    else:
        report_content.append(f"  - Random search finished close or ahead ({rand_search.best_score_:.4f} vs {study.best_value:.4f}) within 30 trials.")
        report_content.append("    TPE's surrogate requires initial warm-up trials before focusing on promising parameter subspaces.")
    report_content.append("\nGuidance:")
    report_content.append("  - For fast models or few hyperparameters, Random Search is easily parallelizable and often sufficient.")
    report_content.append("  - For high-dimensional hyperparameter spaces or expensive objective evaluations, Bayesian Optimization excels.")

    text_report = "\n".join(report_content)
    report_path = config.REPORTS_DIR / "bayesian_optimization_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text_report)
    if config.BAYESOPT_DIR != config.REPORTS_DIR:
        with open(config.BAYESOPT_DIR / "bayesian_optimization_report.txt", "w", encoding="utf-8") as f:
            f.write(text_report)

    print(f"\n[DONE] Bayesian optimization pipeline completed in {t_rand + t_bayes:.1f}s.")
    print(f"       Plots: {plot_running_path.name}, {plot_trials_path.name}")
    print(f"       Report: {report_path.name}")

if __name__ == "__main__":
    run_bayesian_optimization()
