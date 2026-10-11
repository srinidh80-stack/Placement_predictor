"""
Confusion Matrix & Decision Threshold Tuning
Course: Machine Learning (25SC2107E) - Module 5: Evaluation & Validation
Reference: Session: Confusion Matrix & Threshold Tuning

Explores how decision threshold reshapes the confusion matrix for an already-trained model:
  1. Default threshold (0.5) baseline confusion matrix and rates
  2. Full threshold sweep (0.01 to 0.99) tracking Precision, Recall, Specificity, Accuracy, and F1
  3. Threshold selection strategies:
     - Optimal F1-score maximization
     - Asymmetric cost-weighted threshold (e.g. False Negative costs 3x False Positive)
     - Target guaranteed Recall floor (e.g. Recall >= 95%)
  4. Diagnostic visualizations:
     - Multi-panel confusion matrix comparison (t=0.3, 0.5, 0.7)
     - Metrics vs Threshold curve
     - Asymmetric Cost vs Threshold curve
"""

from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.data_utils import ensure_ready_splits

# Ensure output directories exist
config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_threshold_data():
    """Load train_ready and val_ready splits."""
    ensure_ready_splits()
    rt = pd.read_csv(config.SPLITS_DIR / "train_ready.csv")
    rv = pd.read_csv(config.SPLITS_DIR / "val_ready.csv")
    cols = [c for c in rt.columns if c not in ["StudentID", "IsAnomaly", "Salary Package", config.TARGET_COLUMN]]
    X_tr, y_tr = rt[cols], rt[config.TARGET_COLUMN].values
    X_va, y_va = rv[cols], rv[config.TARGET_COLUMN].values
    return {"X_tr": X_tr, "y_tr": y_tr, "X_va": X_va, "y_va": y_va, "cols": cols}


def train_eval_probabilities(X_tr, y_tr, X_va, random_state=42):
    """Fit LightGBM model and predict validation probabilities."""
    model = LGBMClassifier(n_estimators=250, learning_rate=0.05, max_depth=4, random_state=random_state, verbose=-1)
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_va)[:, 1]


def sweep_thresholds(y_va, proba, cost_fn=3, cost_fp=1, recall_floor=0.95):
    """Compute all evaluation metrics and asymmetric cost across fine threshold grid."""
    thresholds = np.linspace(0.01, 0.99, 197)
    metrics_by_t = {"precision": [], "recall": [], "f1": [], "accuracy": [], "specificity": []}
    total_cost = []

    for t in thresholds:
        pred = (proba >= t).astype(int)
        tn_, fp_, fn_, tp_ = confusion_matrix(y_va, pred, labels=[0, 1]).ravel()
        p_ = tp_ / (tp_ + fp_) if (tp_ + fp_) else 0.0
        r_ = tp_ / (tp_ + fn_) if (tp_ + fn_) else 0.0
        spec_ = tn_ / (tn_ + fp_) if (tn_ + fp_) else 0.0
        acc_ = (tp_ + tn_) / len(y_va)
        f1_ = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) else 0.0

        metrics_by_t["precision"].append(p_)
        metrics_by_t["recall"].append(r_)
        metrics_by_t["specificity"].append(spec_)
        metrics_by_t["accuracy"].append(acc_)
        metrics_by_t["f1"].append(f1_)
        total_cost.append(cost_fn * fn_ + cost_fp * fp_)

    for k in metrics_by_t:
        metrics_by_t[k] = np.array(metrics_by_t[k])
    total_cost = np.array(total_cost)

    best_f1_idx = int(np.argmax(metrics_by_t["f1"]))
    t_f1 = float(thresholds[best_f1_idx])

    best_cost_idx = int(np.argmin(total_cost))
    t_cost = float(thresholds[best_cost_idx])

    ok = np.where(metrics_by_t["recall"] >= recall_floor)[0]
    t_recall = float(thresholds[ok[-1]]) if len(ok) else float(thresholds[0])

    return {
        "thresholds": thresholds,
        "metrics": metrics_by_t,
        "total_cost": total_cost,
        "t_f1": t_f1,
        "best_f1_idx": best_f1_idx,
        "t_cost": t_cost,
        "best_cost_idx": best_cost_idx,
        "t_recall": t_recall,
        "ok": ok
    }


def generate_threshold_diagrams(y_va, proba, sweep_res, cost_fn=3, cost_fp=1, recall_floor=0.95):
    """Generate diagnostic plots and save to config.PLOTS_DIR."""
    plots = {}
    thresholds = sweep_res["thresholds"]
    metrics_by_t = sweep_res["metrics"]
    total_cost = sweep_res["total_cost"]
    t_f1 = sweep_res["t_f1"]
    t_cost = sweep_res["t_cost"]
    t_recall = sweep_res["t_recall"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, t, label in zip(axes, [0.3, 0.5, 0.7],
                            ["Low threshold (0.3)\nfavors recall", "Default (0.5)\nbalanced", "High threshold (0.7)\nfavors precision"]):
        pred = (proba >= t).astype(int)
        cm = confusion_matrix(y_va, pred, labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pred: Not Placed", "Pred: Placed"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Actual: Not Placed", "Actual: Placed"])
        ax.set_title(label, fontsize=10, fontweight="bold")
    plt.suptitle("Confusion Matrix Dynamics Across Decision Thresholds", fontsize=12, fontweight="bold")
    plt.tight_layout()
    p1 = config.PLOTS_DIR / "confusion_matrices_by_threshold.png"
    plt.savefig(p1, dpi=150)
    plt.close()
    plots["confusion_matrices"] = p1.name

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, style in [("precision", "-"), ("recall", "-"), ("f1", "-"), ("accuracy", "--"), ("specificity", "--")]:
        ax.plot(thresholds, metrics_by_t[name], style, label=name.capitalize())
    ax.axvline(t_f1, color="#6E2436", linestyle=":", linewidth=1.5, label=f"Max F1 ({t_f1:.2f})")
    ax.axvline(t_cost, color="#D4A017", linestyle=":", linewidth=1.5, label=f"Cost-weighted ({t_cost:.2f})")
    ax.axvline(t_recall, color="#2E7D32", linestyle=":", linewidth=1.5, label=f"95% recall floor ({t_recall:.2f})")
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1, label="Default (0.50)")
    ax.set_xlabel("Decision Threshold", fontweight="bold")
    ax.set_ylabel("Metric Value", fontweight="bold")
    ax.set_title("Performance Metrics vs Decision Threshold", fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    p2 = config.PLOTS_DIR / "metrics_vs_threshold.png"
    plt.savefig(p2, dpi=150)
    plt.close()
    plots["metrics_vs_threshold"] = p2.name

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, total_cost, color="#C0392B", linewidth=2)
    ax.axvline(t_cost, color="#D4A017", linestyle="--", label=f"Minimum cost ({t_cost:.2f})")
    ax.axvline(0.5, color="gray", linestyle=":", label="Default (0.50)")
    ax.set_xlabel("Decision Threshold", fontweight="bold")
    ax.set_ylabel(f"Total Cost ({cost_fn}x per False Negative + {cost_fp}x per False Positive)", fontweight="bold")
    ax.set_title("Cost-Weighted Decision Boundary Optimization", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    p3 = config.PLOTS_DIR / "cost_curve.png"
    plt.savefig(p3, dpi=150)
    plt.close()
    plots["cost_curve"] = p3.name

    return plots


def save_threshold_report(y_va, pred_default, sweep_res, cost_fn, cost_fp, recall_floor):
    """Write comprehensive diagnostic text report to Output/Report/."""
    report_path = config.REPORTS_DIR / "threshold_tuning_report.txt"
    tn, fp, fn, tp = confusion_matrix(y_va, pred_default, labels=[0, 1]).ravel()
    t_f1 = sweep_res["t_f1"]
    t_cost = sweep_res["t_cost"]
    t_recall = sweep_res["t_recall"]
    total_cost = sweep_res["total_cost"]
    best_cost_idx = sweep_res["best_cost_idx"]
    cost_at_default = cost_fn * fn + cost_fp * fp
    metrics_by_t = sweep_res["metrics"]
    ok = sweep_res["ok"]

    with open(report_path, "w") as f:
        f.write("CONFUSION MATRIX & DECISION THRESHOLD TUNING REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Evaluated on {len(y_va):,} validation rows ({y_va.mean():.1%} placed base rate).\n\n")
        f.write(f"1. Default Threshold (0.50):\n")
        f.write(f"   TP={tp}  FP={fp}  FN={fn}  TN={tn}\n")
        f.write(f"   Precision={precision_score(y_va, pred_default):.4f}  Recall={recall_score(y_va, pred_default):.4f}  "
                f"F1={f1_score(y_va, pred_default):.4f}  Accuracy={accuracy_score(y_va, pred_default):.4f}\n\n")

        f.write(f"2. Maximize F1-Score Strategy:\n")
        f.write(f"   Optimal threshold: {t_f1:.3f} (Achieved F1: {metrics_by_t['f1'][sweep_res['best_f1_idx']]:.4f})\n\n")

        f.write(f"3. Asymmetric Cost Minimization (FN cost = {cost_fn}x, FP cost = {cost_fp}x):\n")
        f.write(f"   Optimal threshold: {t_cost:.3f}\n")
        f.write(f"   Cost at optimal: {total_cost[best_cost_idx]:.0f} vs {cost_at_default:.0f} at default "
                f"({(1 - total_cost[best_cost_idx] / cost_at_default) * 100:+.1f}% savings)\n\n")

        f.write(f"4. Guaranteed Recall Floor (Recall >= {recall_floor*100:.0f}%):\n")
        f.write(f"   Optimal threshold: {t_recall:.3f} "
                f"(Recall: {metrics_by_t['recall'][ok[-1] if len(ok) else 0]:.4f}, "
                f"Precision: {metrics_by_t['precision'][ok[-1] if len(ok) else 0]:.4f})\n\n")

        f.write("Takeaway & Production Guidance:\n")
        f.write("  - 0.50 is purely an arbitrary statistical convention, not an optimal decision rule.\n")
        f.write("  - Changing thresholds calibrates real-world utility without needing to retrain models.\n")
    return str(report_path)


def run_threshold_tuning_pipeline(cost_fn=3, cost_fp=1, recall_floor=0.95, random_state=42):
    """End-to-end execution of confusion matrix threshold tuning."""
    print("=" * 80)
    print("  CONFUSION MATRIX & THRESHOLD TUNING PIPELINE")
    print(f"  Target: PlacementStatus | Reference: LightGBM Classifier")
    print("=" * 80)
    start_time = time.perf_counter()

    data = load_threshold_data()
    X_tr, y_tr, X_va, y_va = data["X_tr"], data["y_tr"], data["X_va"], data["y_va"]

    print("\n[1/4] Training model & computing validation probabilities...")
    proba = train_eval_probabilities(X_tr, y_tr, X_va, random_state=random_state)
    pred_default = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_va, pred_default, labels=[0, 1]).ravel()
    print(f"      Default t=0.5: TP={tp} | FP={fp} | FN={fn} | TN={tn}")
    print(f"      Accuracy={accuracy_score(y_va, pred_default):.4f} | F1={f1_score(y_va, pred_default):.4f}")

    print("\n[2/4] Sweeping threshold space (197 intervals from 0.01 to 0.99)...")
    sweep_res = sweep_thresholds(y_va, proba, cost_fn=cost_fn, cost_fp=cost_fp, recall_floor=recall_floor)
    print(f"      Optimal Max F1 Threshold: {sweep_res['t_f1']:.3f}")
    print(f"      Cost-Weighted Optimal Threshold: {sweep_res['t_cost']:.3f}")
    print(f"      Guaranteed Recall Floor Threshold: {sweep_res['t_recall']:.3f}")

    print("\n[3/4] Generating diagnostic plots...")
    plots = generate_threshold_diagrams(y_va, proba, sweep_res, cost_fn=cost_fn, cost_fp=cost_fp, recall_floor=recall_floor)

    print("\n[4/4] Saving threshold tuning report...")
    rep_path = save_threshold_report(y_va, pred_default, sweep_res, cost_fn, cost_fp, recall_floor)

    elapsed = round(time.perf_counter() - start_time, 2)
    print(f"\n[DONE] Threshold tuning pipeline completed in {elapsed}s.")
    print(f"       Plots: {list(plots.values())}")
    print(f"       Report: {Path(rep_path).name}")

    return {
        "sweep_res": sweep_res,
        "plots": plots,
        "report_path": rep_path
    }


if __name__ == "__main__":
    run_threshold_tuning_pipeline()
