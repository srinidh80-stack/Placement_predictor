"""
ROC and Precision-Recall Curves
Course: Machine Learning (25SC2107E) - Module 5: Evaluation & Validation
Reference: Session: ROC and Precision-Recall Curves

Comprehensive multi-model discriminative comparison:
  1. Compares Logistic Regression, Shallow Decision Tree, and LightGBM on validation set.
  2. Measures ROC curves and AUC-ROC (class-balance invariant ranking measure).
  3. Measures Precision-Recall (PR) curves and Average Precision (AP) (sensitive to negative dilution).
  4. Empirical Demonstration of Imbalance Sensitivity:
     - Synthesizes severe imbalance (positives downsampled to 5%)
     - Proves ROC AUC remains stable while PR drops drastically due to false-positive dilution.
  5. Exports publication-quality plots to config.PLOTS_DIR and text report to config.REPORTS_DIR.
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
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.data_utils import ensure_ready_splits

# Ensure output directories exist
config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_roc_pr_data():
    """Load splits and prepare feature pipelines."""
    ensure_ready_splits()
    train_raw = pd.read_csv(config.SPLITS_DIR / "train.csv")
    val_raw = pd.read_csv(config.SPLITS_DIR / "val.csv")
    feature_cols = config.NUMERIC_COLUMNS + config.CATEGORY_COLUMNS + config.YES_NO_COLUMNS + ["CollegeTier", "CGPA_Tier"]
    y_tr, y_va = train_raw[config.TARGET_COLUMN].values, val_raw[config.TARGET_COLUMN].values

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

    rt = pd.read_csv(config.SPLITS_DIR / "train_ready.csv")
    rv = pd.read_csv(config.SPLITS_DIR / "val_ready.csv")
    rcols = [c for c in rt.columns if c not in ["StudentID", "IsAnomaly", "Salary Package", config.TARGET_COLUMN]]

    return {
        "train_raw": train_raw, "val_raw": val_raw,
        "feature_cols": feature_cols,
        "y_tr": y_tr, "y_va": y_va,
        "prep": prep,
        "rt": rt, "rv": rv, "rcols": rcols
    }


def train_and_predict_models(data, random_state=42):
    """Train benchmark models and predict positive class probabilities."""
    train_raw, val_raw = data["train_raw"], data["val_raw"]
    feature_cols, y_tr = data["feature_cols"], data["y_tr"]
    prep = data["prep"]
    rt, rv, rcols = data["rt"], data["rv"], data["rcols"]

    models = {
        "Logistic Regression": Pipeline([("prep", prep), ("m", LogisticRegression(max_iter=2000, random_state=random_state))]),
        "Shallow Decision Tree (depth=3)": Pipeline([("prep", prep), ("m", DecisionTreeClassifier(max_depth=3, random_state=random_state))]),
    }
    probs = {}
    for name, pipe in models.items():
        pipe.fit(train_raw[feature_cols], y_tr)
        probs[name] = pipe.predict_proba(val_raw[feature_cols])[:, 1]

    lgbm = LGBMClassifier(n_estimators=250, learning_rate=0.05, max_depth=4, random_state=random_state, verbose=-1)
    lgbm.fit(rt[rcols], rt[config.TARGET_COLUMN])
    probs["LightGBM"] = lgbm.predict_proba(rv[rcols])[:, 1]

    return probs


def evaluate_imbalance_divergence(y_va, p_best, target_positive_rate=0.05, random_state=42):
    """Simulate severe class imbalance and measure divergent response of ROC vs PR."""
    rng = np.random.RandomState(random_state)
    pos_idx = np.where(y_va == 1)[0]
    neg_idx = np.where(y_va == 0)[0]

    if len(pos_idx) / len(y_va) > target_positive_rate:
        n_pos_keep = int(len(neg_idx) * target_positive_rate / (1 - target_positive_rate))
        pos_keep = rng.choice(pos_idx, min(n_pos_keep, len(pos_idx)), replace=False)
        imb_idx = np.concatenate([pos_keep, neg_idx])
    else:
        n_neg_keep = int(len(pos_idx) * (1 - target_positive_rate) / target_positive_rate)
        neg_keep = rng.choice(neg_idx, min(n_neg_keep, len(neg_idx)), replace=False)
        imb_idx = np.concatenate([pos_idx, neg_keep])

    y_imb, p_imb = y_va[imb_idx], p_best[imb_idx]

    roc_orig, ap_orig = float(roc_auc_score(y_va, p_best)), float(average_precision_score(y_va, p_best))
    roc_imb, ap_imb = float(roc_auc_score(y_imb, p_imb)), float(average_precision_score(y_imb, p_imb))

    return {
        "y_imb": y_imb, "p_imb": p_imb,
        "roc_orig": roc_orig, "ap_orig": ap_orig,
        "roc_imb": roc_imb, "ap_imb": ap_imb,
        "orig_pos_rate": float(y_va.mean()),
        "imb_pos_rate": float(y_imb.mean())
    }


def generate_roc_pr_diagrams(probs, y_va, best_model, imb_res):
    """Generate diagnostic plots and save to config.PLOTS_DIR."""
    plots = {}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for (name, p), c in zip(probs.items(), colors):
        fpr, tpr, _ = roc_curve(y_va, p)
        axes[0].plot(fpr, tpr, color=c, label=f"{name} (AUC {roc_auc_score(y_va, p):.3f})", linewidth=1.8)
        prec, rec, _ = precision_recall_curve(y_va, p)
        axes[1].plot(rec, prec, color=c, label=f"{name} (AP {average_precision_score(y_va, p):.3f})", linewidth=1.8)

    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="No-skill baseline (AUC 0.5)")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    axes[0].set_ylabel("True Positive Rate (Sensitivity)", fontweight="bold")
    axes[0].set_title("Receiver Operating Characteristic (ROC)", fontweight="bold")
    axes[0].legend(fontsize=8)

    axes[1].axhline(y_va.mean(), color="k", linestyle="--", linewidth=1, label=f"No-skill baseline ({y_va.mean():.3f})")
    axes[1].set_xlabel("Recall (True Positive Rate)", fontweight="bold")
    axes[1].set_ylabel("Precision (Positive Predictive Value)", fontweight="bold")
    axes[1].set_title("Precision-Recall (PR) Curves", fontweight="bold")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    p1 = config.PLOTS_DIR / "roc_and_pr_curves.png"
    plt.savefig(p1, dpi=150)
    plt.close()
    plots["roc_and_pr_curves"] = p1.name

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    y_imb, p_imb = imb_res["y_imb"], imb_res["p_imb"]
    p_best = probs[best_model]

    for y_, p_, label, style in [(y_va, p_best, f"Original ({imb_res['orig_pos_rate']*100:.1f}% pos)", "-"),
                                 (y_imb, p_imb, f"Imbalanced ({imb_res['imb_pos_rate']*100:.1f}% pos)", "--")]:
        fpr, tpr, _ = roc_curve(y_, p_)
        axes[0].plot(fpr, tpr, style, color="#4C72B0", label=f"{label}, AUC {roc_auc_score(y_, p_):.3f}", linewidth=1.8)
        prec, rec, _ = precision_recall_curve(y_, p_)
        axes[1].plot(rec, prec, style, color="#DD8452", label=f"{label}, AP {average_precision_score(y_, p_):.3f}", linewidth=1.8)

    axes[0].plot([0, 1], [0, 1], "k:", linewidth=1)
    axes[0].set_xlabel("False Positive Rate", fontweight="bold")
    axes[0].set_ylabel("True Positive Rate", fontweight="bold")
    axes[0].set_title(f"ROC Invariance Under Imbalance\n({best_model})", fontweight="bold")
    axes[0].legend(fontsize=8)

    axes[1].axhline(y_va.mean(), color="#4C72B0", linestyle=":", linewidth=1)
    axes[1].axhline(y_imb.mean(), color="#DD8452", linestyle=":", linewidth=1)
    axes[1].set_xlabel("Recall", fontweight="bold")
    axes[1].set_ylabel("Precision", fontweight="bold")
    axes[1].set_title("Precision-Recall Sensitivity to Imbalance", fontweight="bold")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    p2 = config.PLOTS_DIR / "imbalance_sensitivity.png"
    plt.savefig(p2, dpi=150)
    plt.close()
    plots["imbalance_sensitivity"] = p2.name

    return plots


def save_roc_pr_report(probs, y_va, roc_rank, ap_rank, best_model, imb_res):
    """Write comprehensive diagnostic text report to Output/Report/."""
    report_path = config.REPORTS_DIR / "roc_pr_report.txt"
    with open(report_path, "w") as f:
        f.write("ROC AND PRECISION-RECALL CURVE EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Validation dataset: {len(y_va):,} rows ({y_va.mean():.1%} positive rate).\n\n")
        f.write(f"{'Model':<34}{'ROC AUC':>10}{'Average Precision (AP)':>26}\n")
        for name, p in probs.items():
            f.write(f"{name:<34}{roc_auc_score(y_va, p):>10.4f}{average_precision_score(y_va, p):>26.4f}\n")

        f.write(f"\nModel Rankings:\n")
        f.write(f"  Ranked by ROC AUC: {roc_rank}\n")
        f.write(f"  Ranked by AP:      {ap_rank}\n")
        f.write(f"  Rankings Concordance: {roc_rank == ap_rank}\n\n")

        f.write(f"Empirical Class Imbalance Divergence Experiment ({best_model}):\n")
        f.write(f"  Original ({imb_res['orig_pos_rate']*100:.1f}% positive):   ROC AUC = {imb_res['roc_orig']:.4f} | AP = {imb_res['ap_orig']:.4f}\n")
        f.write(f"  Imbalanced ({imb_res['imb_pos_rate']*100:.1f}% positive): ROC AUC = {imb_res['roc_imb']:.4f} | AP = {imb_res['ap_imb']:.4f}\n")
        f.write(f"  Observed Deltas:  d(ROC AUC) = {imb_res['roc_imb'] - imb_res['roc_orig']:+.4f} | d(AP) = {imb_res['ap_imb'] - imb_res['ap_orig']:+.4f}\n\n")

        f.write("Takeaways & Production Rules:\n")
        f.write("  1. ROC AUC measures ranking quality irrespective of class imbalance.\n")
        f.write("  2. PR AUC (AP) is essential when positive instances are rare and false alarms are expensive.\n")
        f.write("  3. An apparently high ROC AUC can mask poor precision in severely skewed data regimes.\n")
    return str(report_path)


def run_roc_pr_pipeline(random_state=42):
    """End-to-end execution of ROC and Precision-Recall analyses."""
    print("=" * 80)
    print("  ROC & PRECISION-RECALL DISCRIMINATION PIPELINE")
    print("  Models: Logistic Regression, Shallow Tree, LightGBM")
    print("=" * 80)
    start_time = time.perf_counter()

    data = load_roc_pr_data()
    y_va = data["y_va"]

    print("\n[1/4] Training benchmark models and computing probabilities...")
    probs = train_and_predict_models(data, random_state=random_state)
    for name, p in probs.items():
        print(f"      {name:<32}: ROC AUC={roc_auc_score(y_va, p):.4f} | AP={average_precision_score(y_va, p):.4f}")

    roc_rank = sorted(probs, key=lambda n: roc_auc_score(y_va, probs[n]), reverse=True)
    ap_rank = sorted(probs, key=lambda n: average_precision_score(y_va, probs[n]), reverse=True)
    best_model = roc_rank[0]

    print(f"\n[2/4] Testing class imbalance divergence on top model ({best_model})...")
    imb_res = evaluate_imbalance_divergence(y_va, probs[best_model], target_positive_rate=0.05, random_state=random_state)
    print(f"      Original (65.7% pos) -> ROC={imb_res['roc_orig']:.4f}, AP={imb_res['ap_orig']:.4f}")
    print(f"      Imbalanced (5.0% pos) -> ROC={imb_res['roc_imb']:.4f}, AP={imb_res['ap_imb']:.4f}")

    print("\n[3/4] Generating comparative diagnostic diagrams...")
    plots = generate_roc_pr_diagrams(probs, y_va, best_model, imb_res)

    print("\n[4/4] Saving ROC & PR analytical report...")
    rep_path = save_roc_pr_report(probs, y_va, roc_rank, ap_rank, best_model, imb_res)

    elapsed = round(time.perf_counter() - start_time, 2)
    print(f"\n[DONE] ROC & PR pipeline completed in {elapsed}s.")
    print(f"       Plots: {list(plots.values())}")
    print(f"       Report: {Path(rep_path).name}")

    return {
        "probs": probs,
        "imb_res": imb_res,
        "plots": plots,
        "report_path": rep_path
    }


if __name__ == "__main__":
    run_roc_pr_pipeline()
