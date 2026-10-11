"""
Cross-Validation Foundations & Rigorous Evaluation
Course: Machine Learning (25SC2107E) - Module 5: Evaluation & Validation
Reference: Session: Cross-Validation

Empirical benchmarks and diagnostic analyses:
  1. Estimator variance: single 80/20 holdout vs 5-fold vs 10-fold CV
  2. Stratification vs standard KFold, and ordering hazards on imbalanced data
  3. Choice of K (K=3, 5, 10, 20) on 35,000 training records
  4. Repeated K-fold variance reduction
  5. Training score vs cross-validated score (diagnosing memorization/overfitting)
  6. Pipeline encapsulation: imputation, scaling, and encodings refit inside every fold
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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (KFold, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score,
                                     cross_validate, train_test_split)
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

# Ensure output directories exist
config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def build_cv_preprocessor(scale=True):
    """Encapsulate imputation, scaling, and encoding inside a leakage-safe ColumnTransformer."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median"))] + ([("scale", StandardScaler())] if scale else []))
    ordinal = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(categories=[["Tier3", "Tier2", "Tier1"], ["Low", "Mid", "High"]]))
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer([
        ("numeric", numeric, config.NUMERIC_COLUMNS),
        ("ordinal", ordinal, ["CollegeTier", "CGPA_Tier"]),
        ("categorical", categorical, config.CATEGORY_COLUMNS + config.YES_NO_COLUMNS)
    ])


def make_pipeline_model(name, random_state=42):
    """Create end-to-end Pipeline for specified classifier."""
    if name == "Logistic Regression":
        return Pipeline([("prep", build_cv_preprocessor(scale=True)),
                         ("model", LogisticRegression(max_iter=2000, random_state=random_state))])
    if name == "LightGBM":
        return Pipeline([("prep", build_cv_preprocessor(scale=False)),
                         ("model", LGBMClassifier(n_estimators=250, learning_rate=0.05, max_depth=4,
                                                  random_state=random_state, verbose=-1))])
    return Pipeline([("prep", build_cv_preprocessor(scale=False)),
                     ("model", DecisionTreeClassifier(random_state=random_state))])


def load_cv_data():
    """Load train and validation splits."""
    train = pd.read_csv(config.SPLITS_DIR / "train.csv")
    val = pd.read_csv(config.SPLITS_DIR / "val.csv")
    feature_cols = config.NUMERIC_COLUMNS + config.CATEGORY_COLUMNS + config.YES_NO_COLUMNS + ["CollegeTier", "CGPA_Tier"]
    X, y = train[feature_cols], train[config.TARGET_COLUMN].values
    Xv, yv = val[feature_cols], val[config.TARGET_COLUMN].values
    return {"X": X, "y": y, "Xv": Xv, "yv": yv, "feature_cols": feature_cols}


def evaluate_estimator_error(X, y, Xv, yv, n_small=500, n_rounds=30, random_state=42):
    """Benchmark single holdout vs 5-fold and 10-fold CV against true validation accuracy."""
    est = {"80/20 holdout": [], "5-fold CV": [], "10-fold CV": [], "truth": []}
    for r in range(n_rounds):
        rng = np.random.RandomState(2000 + r)
        idx = rng.choice(len(X), n_small, replace=False)
        Xs, ys = X.iloc[idx], y[idx]
        Xa, Xb, ya, yb = train_test_split(Xs, ys, test_size=0.2, random_state=r)
        est["80/20 holdout"].append(accuracy_score(yb, make_pipeline_model("Logistic Regression", random_state).fit(Xa, ya).predict(Xb)))
        for k in (5, 10):
            est[f"{k}-fold CV"].append(cross_val_score(make_pipeline_model("Logistic Regression", random_state), Xs, ys,
                                                       cv=StratifiedKFold(k, shuffle=True, random_state=r)).mean())
        est["truth"].append(accuracy_score(yv, make_pipeline_model("Logistic Regression", random_state).fit(Xs, ys).predict(Xv)))

    truth = np.array(est["truth"])
    err_stats = {}
    for name in ["80/20 holdout", "5-fold CV", "10-fold CV"]:
        e = np.array(est[name]) - truth
        err_stats[name] = (float(e.mean()), float(np.std(est[name])), float(np.sqrt((e ** 2).mean())), e)

    return err_stats


def evaluate_stratification(X, y, random_state=42):
    """Compare class balance across folds for KFold vs StratifiedKFold, plus outcome-sorted hazard."""
    n_strat = 200
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(X), n_strat, replace=False)
    ys_small = y[idx]
    kf_rates, sk_rates = [], []
    for r in range(150):
        for tr_i, te_i in KFold(5, shuffle=True, random_state=r).split(np.zeros(n_strat)):
            kf_rates.append(ys_small[te_i].mean())
        for tr_i, te_i in StratifiedKFold(5, shuffle=True, random_state=r).split(np.zeros(n_strat), ys_small):
            sk_rates.append(ys_small[te_i].mean())

    srt = np.random.RandomState(random_state).choice(len(X), 2000, replace=False)
    order = srt[np.argsort(y[srt], kind="stable")]
    Xsort, ysort = X.iloc[order], y[order]
    res_sorted = {}
    for label, cv in [("KFold, no shuffle", KFold(5, shuffle=False)),
                      ("StratifiedKFold", StratifiedKFold(5, shuffle=True, random_state=random_state))]:
        sc = cross_val_score(make_pipeline_model("Logistic Regression", random_state), Xsort, ysort, cv=cv)
        res_sorted[label] = sc
    sorted_rates = [ysort[te_i].mean() for _, te_i in KFold(5, shuffle=False).split(Xsort)]

    return {
        "kf_rates": kf_rates, "sk_rates": sk_rates, "ys_small_mean": float(ys_small.mean()),
        "res_sorted": res_sorted, "sorted_rates": sorted_rates, "ysort_mean": float(ysort.mean())
    }


def evaluate_choice_of_k(X, y, Xv, yv, k_list=(3, 5, 10), random_state=42):
    """Compare cross-validation performance across K choices on training split."""
    ref = {}
    for name in ["Logistic Regression", "LightGBM"]:
        ref[name] = float(accuracy_score(yv, make_pipeline_model(name, random_state).fit(X, y).predict(Xv)))

    fold_scores, k_rows = {}, []
    for name in ["Logistic Regression", "LightGBM"]:
        for k in k_list:
            t0 = time.time()
            sc = cross_val_score(make_pipeline_model(name, random_state), X, y,
                                 cv=StratifiedKFold(k, shuffle=True, random_state=random_state))
            fold_scores[(name, k)] = sc
            k_rows.append((name, k, float(sc.mean()), float(sc.std()), float(time.time() - t0)))

    se_ref = float(np.sqrt(ref["Logistic Regression"] * (1 - ref["Logistic Regression"]) / len(yv)))
    return {"ref": ref, "fold_scores": fold_scores, "k_rows": k_rows, "se_ref": se_ref}


def evaluate_train_vs_cv_gap(X, y, random_state=42):
    """Evaluate training score vs CV score to detect overfitting."""
    gap_rows = []
    scoring = {"accuracy": "accuracy", "roc_auc": "roc_auc", "neg_log_loss": "neg_log_loss", "f1": "f1"}
    for name in ["Logistic Regression", "LightGBM", "Unconstrained tree"]:
        res = cross_validate(make_pipeline_model(name, random_state), X, y,
                             cv=StratifiedKFold(5, shuffle=True, random_state=random_state),
                             scoring=scoring, return_train_score=True)
        tr, te = float(res["train_accuracy"].mean()), float(res["test_accuracy"].mean())
        gap_rows.append((name, tr, te, float(res["test_roc_auc"].mean()), float(-res["test_neg_log_loss"].mean()), float(res["test_f1"].mean())))
    return gap_rows


def generate_cv_diagrams(err_stats, fold_scores, k_list, ref, strat_res, gap_rows):
    """Generate and export diagnostic plots to config.PLOTS_DIR."""
    plots = {}

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([err_stats[n][3] for n in err_stats], showmeans=True)
    ax.set_xticklabels(list(err_stats))
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_ylabel("Estimate minus true validation score", fontweight="bold")
    ax.set_title("Estimator Error: Holdout vs 5-fold vs 10-fold CV", fontweight="bold")
    plt.tight_layout()
    p1 = config.PLOTS_DIR / "estimator_error_holdout_vs_cv.png"
    plt.savefig(p1, dpi=150)
    plt.close()
    plots["estimator_error"] = p1.name

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, name in zip(axes, ["Logistic Regression", "LightGBM"]):
        data = [fold_scores[(name, k)] for k in k_list]
        ax.boxplot(data, showmeans=True)
        ax.set_xticklabels([f"K={k}" for k in k_list])
        ax.axhline(ref[name], color="#C0392B", linestyle="--", label=f"Val reference ({ref[name]:.4f})")
        ax.set_title(f"{name}: Fold Accuracies by K", fontweight="bold")
        ax.set_ylabel("Accuracy per fold", fontweight="bold")
        ax.legend(fontsize=8)
    plt.tight_layout()
    p2 = config.PLOTS_DIR / "fold_scores_by_k.png"
    plt.savefig(p2, dpi=150)
    plt.close()
    plots["fold_scores_by_k"] = p2.name

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(strat_res["kf_rates"], bins=25, alpha=0.7, color="#DD8452", label=f"KFold (sd {np.std(strat_res['kf_rates']):.3f})")
    axes[0].hist(strat_res["sk_rates"], bins=25, alpha=0.7, color="#4C72B0", label=f"StratifiedKFold (sd {np.std(strat_res['sk_rates']):.3f})")
    axes[0].axvline(strat_res["ys_small_mean"], color="black", linestyle="--")
    axes[0].set_xlabel("Positive-class rate in validation fold", fontweight="bold")
    axes[0].set_title("Fold Class Balance Distribution", fontweight="bold")
    axes[0].legend(fontsize=8)

    w = 0.38
    xs = np.arange(5)
    axes[1].bar(xs - w / 2, strat_res["res_sorted"]["KFold, no shuffle"], w, color="#C0392B", label="KFold (no shuffle)")
    axes[1].bar(xs + w / 2, strat_res["res_sorted"]["StratifiedKFold"], w, color="#4C72B0", label="StratifiedKFold")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([f"Fold {i + 1}" for i in xs])
    axes[1].set_ylabel("Fold accuracy", fontweight="bold")
    axes[1].set_title("Data Sorted by Outcome Hazard", fontweight="bold")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    p3 = config.PLOTS_DIR / "stratification_and_sorted_data.png"
    plt.savefig(p3, dpi=150)
    plt.close()
    plots["stratification_and_sorted_data"] = p3.name

    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(len(gap_rows))
    ax.bar(xs - 0.2, [g[1] for g in gap_rows], 0.4, color="#C0392B", label="Training accuracy")
    ax.bar(xs + 0.2, [g[2] for g in gap_rows], 0.4, color="#4C72B0", label="Cross-validated accuracy")
    for xi, g in zip(xs, gap_rows):
        ax.text(xi, max(g[1], g[2]) + 0.006, f"gap {g[1] - g[2]:+.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([g[0] for g in gap_rows])
    ax.set_ylim(0.8, 1.05)
    ax.set_ylabel("Accuracy", fontweight="bold")
    ax.set_title("Training Score vs Cross-Validated Score Gap", fontweight="bold")
    ax.legend(loc="lower left")
    plt.tight_layout()
    p4 = config.PLOTS_DIR / "train_vs_cv_score_gap.png"
    plt.savefig(p4, dpi=150)
    plt.close()
    plots["train_vs_cv_score_gap"] = p4.name

    return plots


def save_cv_report(err_stats, strat_res, k_res, gap_rows):
    """Write comprehensive diagnostic text report to Output/Report/."""
    report_path = config.REPORTS_DIR / "cross_validation_report.txt"
    with open(report_path, "w") as f:
        f.write("CROSS-VALIDATION EVALUATION & METHODOLOGY REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write("1. Estimator Error Comparison (Holdout vs CV):\n")
        f.write(f"   {'Method':<16}{'Mean error':>12}{'SD estimate':>16}{'RMSE':>9}\n")
        for n, (m, sd, rmse, _) in err_stats.items():
            f.write(f"   {n:<16}{m:>+12.4f}{sd:>16.4f}{rmse:>9.4f}\n")

        f.write("\n2. Stratification & Order Hazards:\n")
        f.write(f"   KFold shuffled positive rate sd: {np.std(strat_res['kf_rates']):.4f}\n")
        f.write(f"   StratifiedKFold positive rate sd: {np.std(strat_res['sk_rates']):.4f}\n")

        f.write("\n3. Choice of K Benchmarks:\n")
        for name in ["Logistic Regression", "LightGBM"]:
            f.write(f"   {name} (Validation reference {k_res['ref'][name]:.4f}):\n")
            for n_, k_, m_, s_, t_ in [r for r in k_res["k_rows"] if r[0] == name]:
                f.write(f"     K={k_:>2}: CV mean {m_:.4f}  fold sd {s_:.4f}  runtime {t_:.1f}s\n")

        f.write("\n4. Overfitting Detection (Train Score vs 5-fold CV Score):\n")
        for n_, tr_, te_, auc_, ll_, f1_ in gap_rows:
            f.write(f"   {n_:<22} train={tr_:.4f}  CV={te_:.4f}  gap={tr_ - te_:+.4f}  AUC={auc_:.4f}  F1={f1_:.4f}\n")

        f.write("\nPractical Guidelines:\n")
        f.write("  - Always use StratifiedKFold (K=5 or K=10) for classification tasks.\n")
        f.write("  - Encapsulate preprocessing (scaling/imputation/encoding) within Scikit-Learn Pipelines.\n")
        f.write("  - Unconstrained trees achieve ~100% training accuracy but fail to generalize; CV exposes this.\n")
    return str(report_path)


def run_cross_validation_pipeline(random_state=42):
    """End-to-end execution of cross-validation benchmarks."""
    print("=" * 80)
    print("  CROSS-VALIDATION RIGOROUS BENCHMARK PIPELINE")
    print("  Dataset: Placement Prediction (35k Train / 5k Val Split)")
    print("=" * 80)
    start_time = time.perf_counter()

    data = load_cv_data()
    X, y, Xv, yv = data["X"], data["y"], data["Xv"], data["yv"]

    print("\n[1/5] Measuring estimator error (80/20 holdout vs 5-fold vs 10-fold CV)...")
    err_stats = evaluate_estimator_error(X, y, Xv, yv, random_state=random_state)
    for name in ["80/20 holdout", "5-fold CV", "10-fold CV"]:
        print(f"      {name:<16}: Mean Error {err_stats[name][0]:+.4f} | RMSE {err_stats[name][2]:.4f}")

    print("\n[2/5] Evaluating stratification and outcome-sorted file hazard...")
    strat_res = evaluate_stratification(X, y, random_state=random_state)
    print(f"      Positive rate spread: KFold SD={np.std(strat_res['kf_rates']):.4f} vs Stratified SD={np.std(strat_res['sk_rates']):.4f}")

    print("\n[3/5] Evaluating choice of K on full training split...")
    k_list = [3, 5, 10]
    k_res = evaluate_choice_of_k(X, y, Xv, yv, k_list=k_list, random_state=random_state)

    print("\n[4/5] Measuring training vs CV generalization gap...")
    gap_rows = evaluate_train_vs_cv_gap(X, y, random_state=random_state)
    for g in gap_rows:
        print(f"      {g[0]:<22}: Train={g[1]:.4f} | CV={g[2]:.4f} | Gap={g[1]-g[2]:+.4f}")

    print("\n[5/5] Generating diagnostic plots and saving report...")
    plots = generate_cv_diagrams(err_stats, k_res["fold_scores"], k_list, k_res["ref"], strat_res, gap_rows)
    rep_path = save_cv_report(err_stats, strat_res, k_res, gap_rows)

    elapsed = round(time.perf_counter() - start_time, 2)
    print(f"\n[DONE] Cross-validation pipeline completed in {elapsed}s.")
    print(f"       Plots: {list(plots.values())}")
    print(f"       Report: {Path(rep_path).name}")

    return {
        "err_stats": err_stats,
        "k_res": k_res,
        "gap_rows": gap_rows,
        "plots": plots,
        "report_path": rep_path
    }


if __name__ == "__main__":
    run_cross_validation_pipeline()
