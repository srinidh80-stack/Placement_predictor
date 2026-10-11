"""
Combining Clustering With a Predictive Model (Feature Augmentation)
Course: Machine Learning (25SC2107E) - Module 4: Unsupervised Learning
Reference: Session: Clustering Feature Augmentation

Uses unsupervised clustering to help a supervised model:
1. Run K-Means on continuous feature space.
2. Augment feature matrices with:
   - Cluster membership (one-hot)
   - Continuous Euclidean distances to each cluster centroid
3. Rigorous Leakage Prevention:
   - K-Means is fitted strictly on the TRAINING set only.
   - Validation rows are assigned with predict() and transform() without seeing PlacementStatus.
4. Paired Bootstrap Verification:
   - 300 bootstrap resamples on validation set with 95% Confidence Intervals
   - Determines whether observed gains are statistically distinguishable from noise.
5. Exports diagnostic plots to config.PLOTS_DIR and text report to config.REPORTS_DIR.
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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.data_utils import ensure_ready_splits

# Ensure output directories exist
config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_augmentation_data():
    """Load raw splits and preprocessed ready splits."""
    ensure_ready_splits()
    raw_tr = pd.read_csv(config.SPLITS_DIR / "train.csv")
    raw_va = pd.read_csv(config.SPLITS_DIR / "val.csv")
    rdy_tr = pd.read_csv(config.SPLITS_DIR / "train_ready.csv")
    rdy_va = pd.read_csv(config.SPLITS_DIR / "val_ready.csv")

    assert (raw_tr.StudentID.values == rdy_tr.StudentID.values).all(), "Train StudentID mismatch"
    assert (raw_va.StudentID.values == rdy_va.StudentID.values).all(), "Validation StudentID mismatch"

    drop_cols = ["StudentID", "IsAnomaly", "Salary Package", config.TARGET_COLUMN]
    base_cols = [c for c in rdy_tr.columns if c not in drop_cols]
    y_tr, y_va = rdy_tr[config.TARGET_COLUMN].values, rdy_va[config.TARGET_COLUMN].values

    # Impute + scale continuous numeric space (fitted on train only)
    imputer = SimpleImputer(strategy="median").fit(raw_tr[config.NUMERIC_COLUMNS])
    scaler = StandardScaler().fit(imputer.transform(raw_tr[config.NUMERIC_COLUMNS]))
    z_tr = scaler.transform(imputer.transform(raw_tr[config.NUMERIC_COLUMNS]))
    z_va = scaler.transform(imputer.transform(raw_va[config.NUMERIC_COLUMNS]))

    return {
        "raw_tr": raw_tr, "raw_va": raw_va,
        "rdy_tr": rdy_tr, "rdy_va": rdy_va,
        "base_cols": base_cols,
        "y_tr": y_tr, "y_va": y_va,
        "z_tr": z_tr, "z_va": z_va
    }


def cluster_features(z_tr, z_va, k, random_state=42):
    """Fit K-Means on train only; return cluster assignments and centroid distances."""
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state).fit(z_tr)
    return {
        "lab_tr": km.labels_,
        "lab_va": km.predict(z_va),
        "dist_tr": km.transform(z_tr),
        "dist_va": km.transform(z_va),
    }


def onehot(labels, k, prefix="cluster"):
    return pd.DataFrame(np.eye(k)[labels], columns=[f"{prefix}_{i}" for i in range(k)])


def build_variant(data_bundle, name, k, random_state=42):
    """Construct (X_train, X_val) for a specified feature-augmentation variant."""
    rdy_tr = data_bundle["rdy_tr"]
    rdy_va = data_bundle["rdy_va"]
    base_cols = data_bundle["base_cols"]
    z_tr = data_bundle["z_tr"]
    z_va = data_bundle["z_va"]

    Xtr = rdy_tr[base_cols].reset_index(drop=True)
    Xva = rdy_va[base_cols].reset_index(drop=True)
    if name == "Baseline":
        return Xtr, Xva

    cf = cluster_features(z_tr, z_va, k, random_state=random_state)
    parts_tr, parts_va = [Xtr], [Xva]

    if name in ("+ cluster label", "+ label + distances"):
        parts_tr.append(onehot(cf["lab_tr"], k, "cluster"))
        parts_va.append(onehot(cf["lab_va"], k, "cluster"))
    if name in ("+ centroid distances", "+ label + distances"):
        cols = [f"dist_c{i}" for i in range(k)]
        parts_tr.append(pd.DataFrame(cf["dist_tr"], columns=cols))
        parts_va.append(pd.DataFrame(cf["dist_va"], columns=cols))

    return pd.concat(parts_tr, axis=1), pd.concat(parts_va, axis=1)


def fit_predict(model_name, Xtr, Xva, y_tr, random_state=42):
    """Train predictive model on augmented training data and predict validation probabilities."""
    if model_name == "Logistic Regression":
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(max_iter=2000, random_state=random_state).fit(sc.transform(Xtr), y_tr)
        return m.predict_proba(sc.transform(Xva))[:, 1]
    m = LGBMClassifier(n_estimators=250, learning_rate=0.05, max_depth=4,
                       random_state=random_state, verbose=-1).fit(Xtr, y_tr)
    return m.predict_proba(Xva)[:, 1]


def compute_metrics(y_true, p):
    return {
        "acc": accuracy_score(y_true, p >= 0.5),
        "auc": roc_auc_score(y_true, p),
        "logloss": log_loss(y_true, p)
    }


def paired_auc_diff(p_new, p_base, y_va, boot_idx):
    diffs = np.array([roc_auc_score(y_va[i], p_new[i]) - roc_auc_score(y_va[i], p_base[i]) for i in boot_idx])
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def generate_augmentation_diagrams(scores, variants, models, main_k, k_values, sweep, boot_rows):
    """Generate and save publication-grade diagnostic plots."""
    plots = {}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(variants))
    w = 0.38
    for j, mdl in enumerate(models):
        aucs = [scores[(mdl, v)]["auc"] for v in variants]
        axes[0].bar(x + (j - 0.5) * w, aucs, w, label=mdl, color=["#4C72B0", "#DD8452"][j])
        for xi, a in zip(x, aucs):
            axes[0].text(xi + (j - 0.5) * w, a + 0.0004, f"{a:.4f}", ha="center", fontsize=7, rotation=90)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([v.replace("+ ", "+\n") for v in variants], fontsize=8)
    lo_all = min(scores[k]["auc"] for k in scores)
    axes[0].set_ylim(lo_all - 0.004, max(scores[k]["auc"] for k in scores) + 0.004)
    axes[0].set_ylabel("Validation AUC", fontweight="bold")
    axes[0].set_title(f"Feature Augmentation With K-Means (K={main_k})", fontweight="bold")
    axes[0].legend()

    for mdl, color in zip(models, ["#4C72B0", "#DD8452"]):
        axes[1].plot(k_values, sweep[mdl], marker="o", color=color, label=f"{mdl} + distances")
        axes[1].axhline(scores[(mdl, "Baseline")]["auc"], color=color, linestyle="--", alpha=0.6,
                        label=f"{mdl} baseline")
    axes[1].set_xlabel("Number of clusters K used to build features", fontweight="bold")
    axes[1].set_ylabel("Validation AUC", fontweight="bold")
    axes[1].set_title("Does More Clusters = More Signal?", fontweight="bold")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    p1 = config.PLOTS_DIR / "augmentation_comparison.png"
    plt.savefig(p1, dpi=150)
    plt.close()
    plots["augmentation_comparison"] = p1.name

    fig, ax = plt.subplots(figsize=(9, 4.5))
    model_abbrev = {"Logistic Regression": "LogReg", "LightGBM": "LightGBM"}
    labels = [f"{model_abbrev[m]}: {v}" for m, v, *_ in boot_rows]
    means = [r[2] for r in boot_rows]
    errs = [[r[2] - r[3] for r in boot_rows], [r[4] - r[2] for r in boot_rows]]
    ax.errorbar(means, range(len(boot_rows)), xerr=errs, fmt="o", color="#6E2436", capsize=4)
    ax.axvline(0, color="gray", linestyle="--")
    ax.set_yticks(range(len(boot_rows)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("AUC improvement over baseline (95% bootstrap CI)", fontweight="bold")
    ax.set_title("Is the Improvement Real? CIs Crossing Zero = Indistinguishable From Noise", fontweight="bold")
    plt.tight_layout()
    p2 = config.PLOTS_DIR / "improvement_confidence_intervals.png"
    plt.savefig(p2, dpi=150)
    plt.close()
    plots["confidence_intervals"] = p2.name

    return plots


def save_augmentation_report(scores, boot_rows, sweep, k_values, main_k, conclusion_lines, models, variants, n_tr, n_va, n_features):
    """Write comprehensive diagnostic text report to Output/Report/."""
    report_path = config.REPORTS_DIR / "cluster_augmentation_report.txt"
    with open(report_path, "w") as f:
        f.write("FEATURE AUGMENTATION WITH CLUSTERING REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Clustering as feature augmentation: K-Means (fitted on {n_tr:,} TRAIN rows only, never\n")
        f.write(f"on validation, never on the label) adds cluster-membership and centroid-distance\n")
        f.write(f"columns to {n_features} base features. Evaluated on {n_va:,} validation rows.\n\n")
        f.write(f"Results at K={main_k}:\n")
        f.write(f"{'Model':<21}{'Feature set':<24}{'Accuracy':>9}{'AUC':>9}{'LogLoss':>9}\n")
        for mdl in models:
            for var in variants:
                s = scores[(mdl, var)]
                f.write(f"{mdl:<21}{var:<24}{s['acc']:>9.4f}{s['auc']:>9.4f}{s['logloss']:>9.4f}\n")
        f.write("\nPaired bootstrap of AUC improvement over Baseline (300 resamples):\n")
        for mdl, var, m, lo, hi, verdict in boot_rows:
            f.write(f"  {mdl:<20}{var:<24}dAUC={m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  -> {verdict}\n")
        f.write("\nValidation AUC vs K (centroid-distance features):\n")
        for mdl in models:
            f.write(f"  {mdl:<20} baseline {scores[(mdl, 'Baseline')]['auc']:.4f} | "
                    + "  ".join(f"K={k}:{a:.4f}" for k, a in zip(k_values, sweep[mdl])) + "\n")
        f.write("\nConclusion:\n")
        for line in conclusion_lines:
            f.write("  " + line + "\n")
        f.write("\nMethodological Insights:\n")
        f.write("  1. Cluster features can only help a model that cannot already represent the\n")
        f.write("     underlying non-linear manifold structure.\n")
        f.write("  2. Confidence intervals that cross zero demonstrate that measured gains are\n")
        f.write("     indistinguishable from sample variance.\n")
        f.write("  3. Leakage safeguard: K-Means centroids fixed strictly at training time.\n")
    return str(report_path)


def run_cluster_augmentation_pipeline(main_k=4, n_boot=300, random_state=42):
    """End-to-end pipeline runner for clustering feature augmentation."""
    print("=" * 80)
    print("  CLUSTERING FEATURE AUGMENTATION PIPELINE")
    print(f"  Target: {config.TARGET_COLUMN} | Evaluator: Logistic Regression & LightGBM")
    print("=" * 80)
    start_time = time.perf_counter()

    data = load_augmentation_data()
    y_va = data["y_va"]
    variants = ["Baseline", "+ cluster label", "+ centroid distances", "+ label + distances"]
    models = ["Logistic Regression", "LightGBM"]
    probs, scores = {}, {}

    print(f"\n[1/4] Evaluating feature augmentation variants at K={main_k}...")
    for mdl in models:
        for var in variants:
            Xtr, Xva = build_variant(data, var, main_k, random_state=random_state)
            p = fit_predict(mdl, Xtr, Xva, data["y_tr"], random_state=random_state)
            probs[(mdl, var)], scores[(mdl, var)] = p, compute_metrics(y_va, p)

    print(f"{'Model':<21}{'Feature set':<24}{'Accuracy':>9}{'AUC':>9}{'LogLoss':>9}")
    for mdl in models:
        for var in variants:
            s = scores[(mdl, var)]
            print(f"{mdl:<21}{var:<24}{s['acc']:>9.4f}{s['auc']:>9.4f}{s['logloss']:>9.4f}")

    print(f"\n[2/4] Conducting paired bootstrap ({n_boot} resamples)...")
    rng = np.random.RandomState(random_state)
    boot_idx = [rng.randint(0, len(y_va), len(y_va)) for _ in range(n_boot)]
    boot_rows = []
    for mdl in models:
        for var in variants[1:]:
            mean_d, lo, hi = paired_auc_diff(probs[(mdl, var)], probs[(mdl, "Baseline")], y_va, boot_idx)
            verdict = "distinguishable from noise" if (lo > 0 or hi < 0) else "NOT distinguishable from noise"
            boot_rows.append((mdl, var, mean_d, lo, hi, verdict))
            print(f"  {mdl:<20}{var:<24}dAUC={mean_d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  -> {verdict}")

    print("\n[3/4] Sweeping K clusters for centroid distance features...")
    k_values = [2, 3, 4, 6, 8, 12]
    sweep = {mdl: [] for mdl in models}
    for k in k_values:
        Xtr, Xva = build_variant(data, "+ centroid distances", k, random_state=random_state)
        for mdl in models:
            sweep[mdl].append(compute_metrics(y_va, fit_predict(mdl, Xtr, Xva, data["y_tr"], random_state=random_state))["auc"])

    conclusion_lines = []
    n_real_gain = sum(1 for r in boot_rows if r[3] > 0)
    n_real_loss = sum(1 for r in boot_rows if r[4] < 0)
    if n_real_gain == 0 and n_real_loss == 0:
        conclusion_lines.append(f"None of the {len(boot_rows)} augmented variants differs from baseline beyond noise.")
        conclusion_lines.append("On this dataset, K-Means augmented features demonstrate a valid NULL RESULT.")
    else:
        conclusion_lines.append(f"{n_real_gain} variant(s) improved and {n_real_loss} degraded beyond noise (CI excludes zero).")

    print("\n[4/4] Generating plots and saving diagnostic report...")
    plots = generate_augmentation_diagrams(scores, variants, models, main_k, k_values, sweep, boot_rows)
    rep_path = save_augmentation_report(
        scores, boot_rows, sweep, k_values, main_k, conclusion_lines,
        models, variants, len(data["rdy_tr"]), len(data["rdy_va"]), len(data["base_cols"])
    )

    elapsed = round(time.perf_counter() - start_time, 2)
    print(f"\n[DONE] Pipeline completed in {elapsed}s.")
    print(f"       Plots: {list(plots.values())}")
    print(f"       Report: {Path(rep_path).name}")

    return {
        "scores": scores,
        "boot_rows": boot_rows,
        "sweep": sweep,
        "plots": plots,
        "report_path": rep_path
    }


if __name__ == "__main__":
    run_cluster_augmentation_pipeline()
