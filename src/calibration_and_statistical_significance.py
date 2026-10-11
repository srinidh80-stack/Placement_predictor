"""Calibration and Statistical Significance Testing.

Module 5 evaluation script covering:
1. Probability Calibration:
   - Evaluates raw LightGBM probabilities vs Isotonic and Sigmoid CalibratedClassifierCV.
   - Computes Brier scores and reliability curves across quantile bins.
2. Statistical Significance:
   - Compares Logistic Regression against LightGBM on identical validation samples.
   - Performs McNemar's test on paired prediction disagreements.
   - Calculates paired bootstrap 95% confidence intervals for ROC AUC differences.
"""

import os
import sys
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
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, accuracy_score
from scipy.stats import chi2
from lightgbm import LGBMClassifier

import config
from src.data_utils import ensure_ready_splits

warnings.filterwarnings("ignore")

def run_calibration_and_significance():
    print("=" * 80)
    print("  CALIBRATION AND STATISTICAL SIGNIFICANCE EVALUATION")
    print("=" * 80)

    ensure_ready_splits()
    RS = config.RANDOM_STATE
    TARGET = config.TARGET_COLUMN

    # =====================================================================
    # Part 1: Calibration
    # =====================================================================
    train_ready_path = config.SPLITS_DIR / "train_ready.csv"
    val_ready_path = config.SPLITS_DIR / "val_ready.csv"

    rt = pd.read_csv(train_ready_path)
    rv = pd.read_csv(val_ready_path)
    rcols = [c for c in rt.columns if c not in ["StudentID", "IsAnomaly", "Salary Package", TARGET]]
    X_tr, y_tr = rt[rcols], rt[TARGET].values
    X_va, y_va = rv[rcols], rv[TARGET].values

    print(f"\n[1/4] Assessing Calibration on {len(X_tr):,} train & {len(X_va):,} validation rows ({y_va.mean():.1%} placed)...")

    lgbm_params = dict(n_estimators=250, learning_rate=0.05, max_depth=4, random_state=RS, verbose=-1, n_jobs=-1)

    print("      Fitting Raw LightGBM model...")
    raw_model = LGBMClassifier(**lgbm_params).fit(X_tr, y_tr)
    raw_probs = raw_model.predict_proba(X_va)[:, 1]

    print("      Fitting Isotonic and Sigmoid CalibratedClassifierCV models (5-fold CV)...")
    isotonic_model = CalibratedClassifierCV(LGBMClassifier(**lgbm_params), method="isotonic", cv=5).fit(X_tr, y_tr)
    sigmoid_model = CalibratedClassifierCV(LGBMClassifier(**lgbm_params), method="sigmoid", cv=5).fit(X_tr, y_tr)
    iso_probs = isotonic_model.predict_proba(X_va)[:, 1]
    sig_probs = sigmoid_model.predict_proba(X_va)[:, 1]

    variants = {
        "Raw LightGBM": raw_probs,
        "Isotonic-calibrated": iso_probs,
        "Sigmoid-calibrated": sig_probs
    }

    brier_scores = {}
    auc_scores = {}
    for name, p in variants.items():
        brier_scores[name] = brier_score_loss(y_va, p)
        auc_scores[name] = roc_auc_score(y_va, p)
        print(f"      {name:<22} Brier: {brier_scores[name]:.4f} | ROC AUC: {auc_scores[name]:.4f}")

    best_calibrated = min(["Isotonic-calibrated", "Sigmoid-calibrated"], key=lambda n: brier_scores[n])
    brier_gain_pct = (brier_scores["Raw LightGBM"] - brier_scores[best_calibrated]) / brier_scores["Raw LightGBM"] * 100
    print(f"      Best calibrated variant: {best_calibrated} ({brier_gain_pct:+.2f}% Brier improvement)")

    # ---------- Reliability curves ----------
    print("\n[2/4] Plotting reliability curves and probability distributions...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = {"Raw LightGBM": "#C0392B", "Isotonic-calibrated": "#4C72B0", "Sigmoid-calibrated": "#55A868"}
    for name, p in variants.items():
        frac_pos, mean_pred = calibration_curve(y_va, p, n_bins=10, strategy="quantile")
        axes[0].plot(mean_pred, frac_pos, "o-", color=colors[name], label=f"{name} (Brier {brier_scores[name]:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    axes[0].set_xlabel("Mean predicted probability (within bin)")
    axes[0].set_ylabel("Actual fraction placed (within bin)")
    axes[0].set_title("Reliability Curve (10 equal-count bins)")
    axes[0].legend(fontsize=8)

    axes[1].hist(raw_probs, bins=30, alpha=0.6, color="#C0392B", label="Raw LightGBM", density=True)
    axes[1].hist(variants[best_calibrated], bins=30, alpha=0.6, color="#4C72B0", label=best_calibrated, density=True)
    axes[1].set_xlabel("Predicted probability")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Predicted Probability Distributions")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plot_calib_path = config.PLOTS_DIR / "reliability_curves.png"
    plt.savefig(plot_calib_path, dpi=130)
    plt.close()

    # =====================================================================
    # Part 2: Statistical significance of a model comparison
    # Logistic Regression vs LightGBM on the exact same validation rows
    # =====================================================================
    print("\n[3/4] Running paired statistical significance tests (Logistic Regression vs LightGBM)...")
    train_raw = pd.read_csv(config.SPLITS_DIR / "train.csv")
    val_raw = pd.read_csv(config.SPLITS_DIR / "val.csv")
    feature_cols = config.NUMERIC_COLUMNS + config.CATEGORY_COLUMNS + config.YES_NO_COLUMNS + ["CollegeTier", "CGPA_Tier"]

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
        ("numeric", numeric, config.NUMERIC_COLUMNS),
        ("ordinal", ordinal, ["CollegeTier", "CGPA_Tier"]),
        ("categorical", categorical, config.CATEGORY_COLUMNS + config.YES_NO_COLUMNS)
    ])
    logreg = Pipeline([("prep", prep), ("m", LogisticRegression(max_iter=2000, random_state=RS))])
    logreg.fit(train_raw[feature_cols], train_raw[TARGET].values)
    logreg_probs = logreg.predict_proba(val_raw[feature_cols])[:, 1]
    logreg_preds = (logreg_probs >= 0.5).astype(int)
    lgbm_preds = (raw_probs >= 0.5).astype(int)

    logreg_acc = accuracy_score(y_va, logreg_preds)
    lgbm_acc = accuracy_score(y_va, lgbm_preds)
    logreg_auc = roc_auc_score(y_va, logreg_probs)
    lgbm_auc = roc_auc_score(y_va, raw_probs)

    print(f"      Logistic Regression: Acc={logreg_acc:.4f} | ROC AUC={logreg_auc:.4f}")
    print(f"      LightGBM:            Acc={lgbm_acc:.4f} | ROC AUC={lgbm_auc:.4f}")

    # McNemar's test
    logreg_correct = (logreg_preds == y_va)
    lgbm_correct = (lgbm_preds == y_va)
    b = int(np.sum(logreg_correct & ~lgbm_correct))
    c = int(np.sum(~logreg_correct & lgbm_correct))
    both_right = int(np.sum(logreg_correct & lgbm_correct))
    both_wrong = int(np.sum(~logreg_correct & ~lgbm_correct))

    mcnemar_stat = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
    mcnemar_p = 1 - chi2.cdf(mcnemar_stat, df=1)

    print(f"      McNemar: b (LogReg only)={b}, c (LightGBM only)={c}")
    print(f"      Chi-square={mcnemar_stat:.3f}, p-value={mcnemar_p:.4g}")

    # Paired bootstrap for AUC difference
    N_BOOT = 2000
    rng = np.random.RandomState(RS)
    n = len(y_va)
    boot_diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.randint(0, n, n)
        y_b = y_va[idx]
        while y_b.min() == y_b.max():
            idx = rng.randint(0, n, n)
            y_b = y_va[idx]
        auc_lgbm_b = roc_auc_score(y_b, raw_probs[idx])
        auc_logreg_b = roc_auc_score(y_b, logreg_probs[idx])
        boot_diffs[i] = auc_lgbm_b - auc_logreg_b

    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    point_diff = lgbm_auc - logreg_auc
    print(f"      Paired Bootstrap (N={N_BOOT}): AUC Gap={point_diff:+.4f}, 95% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]")

    # ---------- Significance plots ----------
    print("\n[4/4] Generating significance plots and diagnostic report...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].bar(["Both correct", "Both wrong", "LogReg only", "LightGBM only"],
                [both_right, both_wrong, b, c], color=["#55A868", "#C0392B", "#DD8452", "#4C72B0"])
    for i, v in enumerate([both_right, both_wrong, b, c]):
        axes[0].text(i, v + len(y_va) * 0.01, f"{v:,}", ha="center", fontsize=9)
    axes[0].set_ylabel("Number of validation rows")
    axes[0].set_title(f"McNemar Contingency Table\n(chi-square {mcnemar_stat:.2f}, p={mcnemar_p:.4g})")

    axes[1].hist(boot_diffs, bins=50, color="#4C72B0", alpha=0.75)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1, label="No difference")
    axes[1].axvline(ci_lo, color="#C0392B", linestyle=":", linewidth=1.5, label=f"95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    axes[1].axvline(ci_hi, color="#C0392B", linestyle=":", linewidth=1.5)
    axes[1].axvline(point_diff, color="#55A868", linewidth=1.5, label=f"Point estimate {point_diff:+.4f}")
    axes[1].set_xlabel("Bootstrap AUC difference (LightGBM - Logistic Regression)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Paired Bootstrap Distribution ({N_BOOT:,} resamples)")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plot_sig_path = config.PLOTS_DIR / "significance_tests.png"
    plt.savefig(plot_sig_path, dpi=130)
    plt.close()

    # Report
    report_content = []
    report_content.append(f"Calibration and statistical significance, {len(X_va):,} validation rows ({y_va.mean():.1%} placed)\n")
    report_content.append("Part 1: Calibration (LightGBM)")
    for name, p in variants.items():
        report_content.append(f"  {name:<22} Brier {brier_scores[name]:.4f}   ROC AUC {auc_scores[name]:.4f}")
    report_content.append(f"  Best-calibrated variant by Brier score: {best_calibrated} ({brier_gain_pct:.1f}% lower Brier than raw)\n")

    report_content.append("Part 2: Statistical significance, Logistic Regression vs LightGBM")
    report_content.append(f"  Logistic Regression: accuracy {logreg_acc:.4f}   ROC AUC {logreg_auc:.4f}")
    report_content.append(f"  LightGBM:             accuracy {lgbm_acc:.4f}   ROC AUC {lgbm_auc:.4f}\n")
    report_content.append(f"  McNemar's test: both correct {both_right:,}, both wrong {both_wrong:,}, LogReg-only-right {b:,}, LightGBM-only-right {c:,}")
    report_content.append(f"  Chi-square (continuity-corrected) {mcnemar_stat:.3f}, p-value {mcnemar_p:.4g}")
    report_content.append(f"  Paired bootstrap AUC difference (LightGBM - LogReg): point {point_diff:+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]\n")

    report_content.append("Guidance:")
    report_content.append("  - Calibration and ranking quality (ROC AUC) evaluate distinct aspects of performance.")
    report_content.append("  - Paired statistical tests (McNemar and paired bootstrap) provide rigorous confidence")
    report_content.append("    intervals when determining whether a model is genuinely superior vs chance variation.")

    text_report = "\n".join(report_content)
    report_path = config.REPORTS_DIR / "calibration_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text_report)
    if config.CALIBRATION_DIR != config.REPORTS_DIR:
        with open(config.CALIBRATION_DIR / "calibration_report.txt", "w", encoding="utf-8") as f:
            f.write(text_report)

    print(f"\n[DONE] Calibration & significance testing completed successfully.")
    print(f"       Plots: {plot_calib_path.name}, {plot_sig_path.name}")
    print(f"       Report: {report_path.name}")

if __name__ == "__main__":
    run_calibration_and_significance()
