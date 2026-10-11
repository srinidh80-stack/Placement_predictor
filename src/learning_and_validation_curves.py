"""Learning Curves and Validation Curves.

Diagnoses bias-variance tradeoff across models and hyperparameter sweeps:
1. Learning curve: train and CV score vs TRAINING SET SIZE (diagnoses need for more data vs bias/variance).
2. Validation curve: train and CV score vs HYPERPARAMETER (diagnoses optimal model complexity).
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
from sklearn.model_selection import learning_curve, validation_curve, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import config
from src.data_utils import ensure_ready_splits

warnings.filterwarnings("ignore")

def run_learning_and_validation_curves():
    print("=" * 80)
    print("  LEARNING & VALIDATION CURVES DIAGNOSTICS")
    print("=" * 80)

    ensure_ready_splits()
    RS = config.RANDOM_STATE
    TARGET = config.TARGET_COLUMN

    train_ready_path = config.SPLITS_DIR / "train_ready.csv"
    rt = pd.read_csv(train_ready_path)
    cols = [c for c in rt.columns if c not in ["StudentID", "IsAnomaly", "Salary Package", TARGET]]
    X, y = rt[cols], rt[TARGET].values

    cv = StratifiedKFold(5, shuffle=True, random_state=RS)
    train_sizes = np.linspace(0.05, 1.0, 10)
    print(f"\n[1/3] Generating learning curves on {len(X):,} training rows (5-fold CV)...")

    models = {
        "Logistic Regression\n(roughly matched complexity)": LogisticRegression(max_iter=1000, random_state=RS),
        "Decision tree, depth=2\n(too simple - high bias)": DecisionTreeClassifier(max_depth=2, random_state=RS),
        "Decision tree, unconstrained\n(too complex - high variance)": DecisionTreeClassifier(random_state=RS),
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    lc_results = {}
    for ax, (name, model) in zip(axes, models.items()):
        sizes, train_scores, cv_scores = learning_curve(model, X, y, cv=cv, train_sizes=train_sizes, n_jobs=-1)
        tr_mean, tr_std = train_scores.mean(axis=1), train_scores.std(axis=1)
        cv_mean, cv_std = cv_scores.mean(axis=1), cv_scores.std(axis=1)
        lc_results[name] = (sizes, tr_mean, cv_mean)
        ax.plot(sizes, tr_mean, "o-", color="#C0392B", label="Training score")
        ax.fill_between(sizes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15, color="#C0392B")
        ax.plot(sizes, cv_mean, "o-", color="#4C72B0", label="CV score")
        ax.fill_between(sizes, cv_mean - cv_std, cv_mean + cv_std, alpha=0.15, color="#4C72B0")
        ax.set_xlabel("Training set size")
        ax.set_title(name, fontsize=10)
        final_gap = tr_mean[-1] - cv_mean[-1]
        ax.text(0.03, 0.03, f"Final gap: {final_gap:+.3f}", transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        print(f"\n      {name.splitlines()[0]}:")
        print(f"        Smallest size ({int(sizes[0])} rows): train {tr_mean[0]:.4f}, CV {cv_mean[0]:.4f}, gap {tr_mean[0]-cv_mean[0]:+.4f}")
        print(f"        Full size ({int(sizes[-1])} rows):     train {tr_mean[-1]:.4f}, CV {cv_mean[-1]:.4f}, gap {final_gap:+.4f}")
        print(f"        CV score improvement:                 {cv_mean[-1]-cv_mean[0]:+.4f}")

    axes[0].set_ylabel("Accuracy")
    axes[0].legend(fontsize=9, loc="lower right")
    plt.suptitle("Learning Curves: Same Data, Three Models in Different Bias/Variance Regimes")
    plt.tight_layout()
    plot_lc_path = config.PLOTS_DIR / "learning_curves_three_models.png"
    plt.savefig(plot_lc_path, dpi=130)
    plt.close()

    print("\n[2/3] Generating validation curve across decision tree max_depth...")
    depth_range = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, None]
    depth_labels = [str(d) for d in depth_range[:-1]] + ["None"]
    train_scores_vc, cv_scores_vc = validation_curve(
        DecisionTreeClassifier(random_state=RS), X, y, param_name="max_depth",
        param_range=depth_range, cv=cv, n_jobs=-1
    )
    tr_vc, cv_vc = train_scores_vc.mean(axis=1), cv_scores_vc.mean(axis=1)
    best_depth_idx = int(np.argmax(cv_vc))

    print(f"      Best depth by CV score: {depth_labels[best_depth_idx]} (CV {cv_vc[best_depth_idx]:.4f})")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x_pos = np.arange(len(depth_labels))
    ax.plot(x_pos, tr_vc, "o-", color="#C0392B", label="Training score")
    ax.plot(x_pos, cv_vc, "o-", color="#4C72B0", label="CV score")
    ax.axvline(best_depth_idx, color="#D4A017", linestyle="--", label=f"Best depth ({depth_labels[best_depth_idx]})")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(depth_labels)
    ax.set_xlabel("max_depth")
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation Curve: Decision Tree Depth (training set size fixed)")
    ax.legend()
    plt.tight_layout()
    plot_vc_path = config.PLOTS_DIR / "validation_curve_tree_depth.png"
    plt.savefig(plot_vc_path, dpi=130)
    plt.close()

    print("\n[3/3] Saving summary diagnostic report...")
    report_content = []
    report_content.append(f"Learning and validation curves, {len(X):,} training rows, 5-fold CV\n")
    for name in models:
        sizes, tr, cv_ = lc_results[name]
        report_content.append(f"{name.splitlines()[0]}:")
        report_content.append(f"  At smallest size ({int(sizes[0])}): train {tr[0]:.4f}, CV {cv_[0]:.4f}, gap {tr[0]-cv_[0]:+.4f}")
        report_content.append(f"  At full size ({int(sizes[-1])}):   train {tr[-1]:.4f}, CV {cv_[-1]:.4f}, gap {tr[-1]-cv_[-1]:+.4f}")
        report_content.append(f"  CV improvement, smallest to largest: {cv_[-1]-cv_[0]:+.4f}\n")

    report_content.append("Validation curve (decision tree depth, full training set):")
    for lbl, tr_, cv__ in zip(depth_labels, tr_vc, cv_vc):
        report_content.append(f"  depth={lbl:<5} train {tr_:.4f}  CV {cv__:.4f}  gap {tr_-cv__:+.4f}")
    report_content.append(f"Best depth by CV score: {depth_labels[best_depth_idx]} (CV {cv_vc[best_depth_idx]:.4f})\n")

    report_content.append("How to tell the two plots apart and when to use each:")
    report_content.append("  - Learning curve: x-axis is training set SIZE, hyperparameters fixed. Answers 'would more data help?'")
    report_content.append("  - Validation curve: x-axis is a HYPERPARAMETER, training set size fixed. Answers 'how complex should")
    report_content.append("    the model be, given the data already in hand?'")
    report_content.append("  - A high-bias model's learning curve plateaus early with train and CV close together and both low.")
    report_content.append("  - A high-variance model's learning curve keeps a wide train-CV gap even at full data size.")

    text_report = "\n".join(report_content)
    report_path = config.REPORTS_DIR / "learning_curves_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text_report)
    if config.LEARNING_CURVE_DIR != config.REPORTS_DIR:
        with open(config.LEARNING_CURVE_DIR / "learning_curves_report.txt", "w", encoding="utf-8") as f:
            f.write(text_report)

    print(f"\n[DONE] Learning and validation curves completed successfully.")
    print(f"       Plots: {plot_lc_path.name}, {plot_vc_path.name}")
    print(f"       Report: {report_path.name}")

if __name__ == "__main__":
    run_learning_and_validation_curves()
