"""
Data Leakage Audit & Demonstration
Course: Machine Learning (25SC2107E) - Module 5: Evaluation & Validation
Reference: Session: Data Leakage

Comprehensive audits and experimental demonstrations of five forms of leakage:
  1. Target leakage: incorporating downstream target information (e.g., Salary Package)
  2. Preprocessing leakage: fitting scalers/imputers before dataset splitting
  3. Feature-selection leakage: selecting predictive features across entire dataset
  4. Duplicate / overlap leakage: train rows contaminating validation/test sets
  5. Selection-set reuse: picking optimal hyperparameters on test evaluation set
"""

from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.data_utils import ensure_ready_splits

# Ensure output directories exist
config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_leakage_data():
    """Load raw and ready splits, ensuring integrity."""
    ensure_ready_splits()
    raw = {n: pd.read_csv(config.SPLITS_DIR / f"{n}.csv") for n in ["train", "val", "test"]}
    rdy = {n: pd.read_csv(config.SPLITS_DIR / f"{n}_ready.csv") for n in ["train", "val", "test"]}

    for n in raw:
        assert (raw[n].StudentID.values == rdy[n].StudentID.values).all(), f"Row order mismatch in {n}"

    drop_cols = ["StudentID", "IsAnomaly", "Salary Package", config.TARGET_COLUMN]
    clean_cols = [c for c in rdy["train"].columns if c not in drop_cols]
    y = {n: rdy[n][config.TARGET_COLUMN].values for n in rdy}

    return {"raw": raw, "rdy": rdy, "clean_cols": clean_cols, "y": y}


def audit_splits(raw):
    """Audit StudentID and feature hash overlap across train, val, and test splits."""
    ids = {n: set(raw[n].StudentID) for n in raw}
    audit_id = {f"{a}/{b}": len(ids[a] & ids[b]) for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}

    feat_cols_raw = [c for c in raw["train"].columns if c != "StudentID"]
    sig = {n: pd.util.hash_pandas_object(raw[n][feat_cols_raw], index=False) for n in raw}
    audit_rows = {f"{a}/{b}": len(set(sig[a]) & set(sig[b])) for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
    train_dups = int(sig["train"].duplicated().sum())

    return {
        "audit_id": audit_id,
        "audit_rows": audit_rows,
        "train_duplicates": train_dups,
        "sig": sig
    }


def audit_single_feature_auc(rdy, y_train):
    """Scan each single feature's AUC to detect suspiciously predictive signals."""
    audit_cols = [c for c in rdy["train"].columns if c not in ("StudentID", config.TARGET_COLUMN)]
    single_auc = {}
    for c in audit_cols:
        a = roc_auc_score(y_train, rdy["train"][c])
        single_auc[c] = max(a, 1 - a)
    top = sorted(single_auc.items(), key=lambda kv: kv[1], reverse=True)[:8]
    suspicious = [c for c, a in single_auc.items() if a > 0.95]
    return {"top_features": top, "suspicious": suspicious, "all_auc": single_auc}


def demo_target_leakage(rdy, clean_cols, y, random_state=42):
    """Demonstrate effect of conditioning on downstream target feature (Salary Package)."""
    res = {}
    for name, cols in [("clean features", clean_cols), ("+ Salary Package", clean_cols + ["Salary Package"])]:
        m = LGBMClassifier(n_estimators=250, learning_rate=0.05, max_depth=4, random_state=random_state, verbose=-1)
        m.fit(rdy["train"][cols], y["train"])
        p = m.predict_proba(rdy["val"][cols])[:, 1]
        res[name] = (float(accuracy_score(y["val"], p >= 0.5)), float(roc_auc_score(y["val"], p)))
    return res


def demo_preprocessing_leakage(raw, y, random_state=42):
    """Demonstrate fitting scaler/imputer on full dataset vs train-only."""
    nc = config.NUMERIC_COLUMNS
    res = {}
    for name in ["fit on train only (correct)", "fit on train+val+test (leaky)"]:
        fit_df = raw["train"][nc] if "correct" in name else pd.concat([raw[n][nc] for n in raw])
        imp = SimpleImputer(strategy="median").fit(fit_df)
        sc = StandardScaler().fit(imp.transform(fit_df))
        tf = lambda d: sc.transform(imp.transform(d[nc]))
        m = LogisticRegression(max_iter=2000, random_state=random_state).fit(tf(raw["train"]), y["train"])
        p = m.predict_proba(tf(raw["val"]))[:, 1]
        res[name] = (float(accuracy_score(y["val"], p >= 0.5)), float(roc_auc_score(y["val"], p)))
    return res


def demo_feature_selection_leakage(random_state=42, n_rounds=25, n_samples=200, n_features=2000, k_best=20):
    """Demonstrate feature selection before cross-validation creating illusion of signal on pure noise."""
    wrong, right = [], []
    cv = StratifiedKFold(5, shuffle=True, random_state=random_state)
    for r in range(n_rounds):
        rng = np.random.RandomState(1000 + r)
        Xn, yn = rng.randn(n_samples, n_features), rng.randint(0, 2, n_samples)
        Xsel = SelectKBest(f_classif, k=k_best).fit_transform(Xn, yn)
        wrong.append(cross_val_score(LogisticRegression(max_iter=1000), Xsel, yn, cv=cv).mean())
        pipe = Pipeline([("select", SelectKBest(f_classif, k=k_best)), ("clf", LogisticRegression(max_iter=1000))])
        right.append(cross_val_score(pipe, Xn, yn, cv=cv).mean())
    return np.array(wrong), np.array(right)


def demo_duplicate_leakage(rdy, clean_cols, y, random_state=42):
    """Demonstrate overfitting and memorization caused by duplicates shared across splits."""
    rng = np.random.RandomState(random_state)
    sub = rng.choice(len(rdy["train"]), 10000, replace=False)
    Xtr_s, ytr_s = rdy["train"][clean_cols].iloc[sub], y["train"][sub]
    rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=random_state, n_jobs=1).fit(Xtr_s, ytr_s)
    Xva, yva = rdy["val"][clean_cols], y["val"]
    clean_acc = float(accuracy_score(yva, rf.predict(Xva)))

    n_dup = int(0.30 * len(Xva) / 0.70)
    dup_idx = rng.choice(len(Xtr_s), n_dup, replace=False)
    Xleaky = pd.concat([Xva, Xtr_s.iloc[dup_idx]])
    yleaky = np.concatenate([yva, ytr_s[dup_idx]])
    leaky_acc = float(accuracy_score(yleaky, rf.predict(Xleaky)))
    dup_only_acc = float(accuracy_score(ytr_s[dup_idx], rf.predict(Xtr_s.iloc[dup_idx])))

    return {
        "clean_acc": clean_acc,
        "leaky_acc": leaky_acc,
        "dup_only_acc": dup_only_acc,
        "n_dup": n_dup
    }


def demo_test_set_reuse(rdy, clean_cols, y, random_state=42, n_rounds=6, n_configs=30):
    """Demonstrate selection optimism when selecting models on validation holdout without fresh test set."""
    Xtr_all, ytr_all = rdy["train"][clean_cols], y["train"]
    Xva, yva = rdy["val"][clean_cols], y["val"]
    Xte, yte = rdy["test"][clean_cols], y["test"]
    sel_sizes = [500, 5000]

    opt = {s: [] for s in sel_sizes}
    opt_raw = {s: [] for s in sel_sizes}
    offsets = {s: [] for s in sel_sizes}
    gap_rows = []

    for rd in range(n_rounds):
        rng = np.random.RandomState(500 + rd)
        tr_idx = rng.choice(len(Xtr_all), 2000, replace=False)
        val_sub = {500: rng.choice(len(Xva), 500, replace=False), 5000: np.arange(len(Xva))}
        val_pred, test_acc = [], []
        for _ in range(n_configs):
            m = LGBMClassifier(n_estimators=int(rng.randint(20, 150)), learning_rate=float(10 ** rng.uniform(-2, -0.7)),
                               num_leaves=int(rng.randint(3, 40)), min_child_samples=int(rng.randint(5, 60)),
                               subsample=0.8, subsample_freq=1, random_state=int(rng.randint(1e6)), verbose=-1)
            m.fit(Xtr_all.iloc[tr_idx], ytr_all[tr_idx])
            val_pred.append(m.predict(Xva))
            test_acc.append(accuracy_score(yte, m.predict(Xte)))

        tests = np.array(test_acc)
        for s in sel_sizes:
            scores_v = np.array([accuracy_score(yva[val_sub[s]], vp[val_sub[s]]) for vp in val_pred])
            b = int(np.argmax(scores_v))
            raw_gap = scores_v[b] - tests[b]
            offset = (scores_v - tests).mean()
            opt_raw[s].append(raw_gap)
            offsets[s].append(offset)
            opt[s].append(raw_gap - offset)
            if s == 500:
                gap_rows.append((scores_v[b] - offset, tests[b]))

    sel500 = float(np.mean([g[0] for g in gap_rows]))
    tst500 = float(np.mean([g[1] for g in gap_rows]))

    return {
        "opt": opt, "opt_raw": opt_raw, "offsets": offsets,
        "sel_sizes": sel_sizes, "sel500": sel500, "tst500": tst500
    }


def generate_leakage_diagrams(summary, wrong_noise, right_noise, sel_sizes, opt):
    """Generate and save publication-grade diagnostic plots."""
    plots = {}

    fig, ax = plt.subplots(figsize=(11, 6))
    ypos = np.arange(len(summary))[::-1]
    for yp, (label, leaky, honest) in zip(ypos, summary):
        ax.barh(yp + 0.2, honest, 0.38, color="#4C72B0", label="Honest score" if yp == ypos[0] else None)
        ax.barh(yp - 0.2, leaky, 0.38, color="#C0392B", label="Leaky (reported) score" if yp == ypos[0] else None)
        ax.text(max(leaky, honest) + 0.008, yp, f"{leaky - honest:+.4f}", va="center", fontsize=9, fontweight="bold")
    ax.set_yticks(ypos)
    ax.set_yticklabels([s[0] for s in summary], fontsize=9)
    ax.set_xlim(0.4, 1.08)
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Accuracy", fontweight="bold")
    ax.set_title("Five Kinds of Leakage: Score Inflation Summary", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2)
    plt.tight_layout()
    p1 = config.PLOTS_DIR / "leakage_inflation_summary.png"
    plt.savefig(p1, dpi=150)
    plt.close()
    plots["leakage_inflation_summary"] = p1.name

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(right_noise, bins=12, alpha=0.7, color="#4C72B0", label=f"Inside CV (mean {right_noise.mean():.3f})")
    axes[0].hist(wrong_noise, bins=12, alpha=0.7, color="#C0392B", label=f"Select-then-CV (mean {wrong_noise.mean():.3f})")
    axes[0].axvline(0.5, color="black", linestyle="--", label="True accuracy (chance)")
    axes[0].set_xlabel("Cross-validated accuracy on pure noise", fontweight="bold")
    axes[0].set_ylabel("Count of repeats", fontweight="bold")
    axes[0].set_title("Feature-Selection Leakage 'Discovers' Noise Signal", fontweight="bold")
    axes[0].legend(fontsize=8)

    axes[1].bar([str(s) for s in sel_sizes], [np.mean(opt[s]) for s in sel_sizes],
                yerr=[np.std(opt[s]) for s in sel_sizes], color="#DD8452", capsize=5)
    axes[1].set_xlabel("Selection set sample size", fontweight="bold")
    axes[1].set_ylabel("Selection effect (winner gap - average gap)", fontweight="bold")
    axes[1].set_title("Selection Optimism Shrinks as Holdout Grows", fontweight="bold")
    plt.tight_layout()
    p2 = config.PLOTS_DIR / "selection_and_noise_demos.png"
    plt.savefig(p2, dpi=150)
    plt.close()
    plots["selection_and_noise_demos"] = p2.name

    return plots


def save_leakage_report(audit_res, top_auc, summary, reuse_res, d_prep):
    """Write comprehensive diagnostic text report to Output/Report/."""
    report_path = config.REPORTS_DIR / "leakage_report.txt"
    with open(report_path, "w") as f:
        f.write("DATA LEAKAGE AUDIT AND DIAGNOSTIC REPORT\n")
        f.write("=" * 80 + "\n")
        f.write("Split integrity audit (StudentID overlap):\n")
        for k, v in audit_res["audit_id"].items():
            f.write(f"  {k}: {v} shared IDs\n")
        f.write(f"Identical feature rows within train: {audit_res['train_duplicates']}\n\n")

        f.write("Single-feature AUC scan, top 8 (flag > 0.95):\n")
        for c, a in top_auc:
            flag = "  <-- SUSPICIOUS" if a > 0.95 else ""
            f.write(f"  {c:<25} {a:.4f}{flag}\n")

        f.write(f"\n{'Case':<48}{'Leaky':>8}{'Honest':>8}{'Inflation':>11}\n")
        for label, leaky, honest in summary:
            f.write(f"{label.replace(chr(10), ' '):<48}{leaky:>8.4f}{honest:>8.4f}{leaky - honest:>+11.4f}\n")

        f.write("\nKey Conclusions & Safeguards:\n")
        f.write("  1. Target leakage is catastrophic; never include downstream variables.\n")
        f.write("  2. Feature selection must occur strictly inside cross-validation folds.\n")
        f.write("  3. Duplicate leakage dramatically inflates metrics for high-capacity models.\n")
        f.write(f"  4. Preprocessing leakage inflation was {d_prep:+.4f} (safe on large datasets, crucial habit).\n")
        f.write("  5. Always maintain an untouched final test split to eliminate model selection bias.\n")
    return str(report_path)


def run_data_leakage_pipeline(random_state=42):
    """End-to-end execution of data leakage audits and experiments."""
    print("=" * 80)
    print("  DATA LEAKAGE AUDITS & EMPIRICAL DEMONSTRATIONS")
    print(f"  Dataset: Placement Prediction (50,000 records)")
    print("=" * 80)
    start_time = time.perf_counter()

    data = load_leakage_data()
    raw, rdy, clean_cols, y = data["raw"], data["rdy"], data["clean_cols"], data["y"]

    print("\n[1/6] Running split integrity audit (IDs and feature signatures)...")
    audit_res = audit_splits(raw)
    for k, v in audit_res["audit_id"].items():
        print(f"      StudentID overlap {k}: {v}")

    print("\n[2/6] Scanning single-feature AUCs for suspicious signal...")
    auc_res = audit_single_feature_auc(rdy, y["train"])
    for c, a in auc_res["top_features"][:5]:
        flag = " <-- SUSPICIOUS (>0.95)" if a > 0.95 else ""
        print(f"      {c:<22} AUC: {a:.4f}{flag}")

    summary = []

    print("\n[3/6] Running Demo 1: Target Leakage (Salary Package)...")
    res1 = demo_target_leakage(rdy, clean_cols, y, random_state=random_state)
    summary.append(("1. Target leakage\n(+ Salary Package)", res1["+ Salary Package"][0], res1["clean features"][0]))
    print(f"      Clean accuracy: {res1['clean features'][0]:.4f} | Leaky accuracy: {res1['+ Salary Package'][0]:.4f}")

    print("\n[4/6] Running Demo 2: Preprocessing Leakage...")
    res2 = demo_preprocessing_leakage(raw, y, random_state=random_state)
    d_prep = res2["fit on train+val+test (leaky)"][0] - res2["fit on train only (correct)"][0]
    summary.append(("2. Preprocessing leakage\n(scaler on all data)", res2["fit on train+val+test (leaky)"][0], res2["fit on train only (correct)"][0]))
    print(f"      Train-only accuracy: {res2['fit on train only (correct)'][0]:.4f} | Leaky: {res2['fit on train+val+test (leaky)'][0]:.4f}")

    print("\n[5/6] Running Demo 3: Feature-Selection Leakage & Demo 4: Duplicates...")
    wrong_noise, right_noise = demo_feature_selection_leakage(random_state=random_state)
    summary.append(("3. Feature-selection leakage\n(pure noise data)", wrong_noise.mean(), right_noise.mean()))

    dup_res = demo_duplicate_leakage(rdy, clean_cols, y, random_state=random_state)
    summary.append(("4. Duplicate leakage\n(30% copies in val)", dup_res["leaky_acc"], dup_res["clean_acc"]))

    print("\n[6/6] Running Demo 5: Selection-Set Reuse & Exporting Reports...")
    reuse_res = demo_test_set_reuse(rdy, clean_cols, y, random_state=random_state)
    summary.append(("5. Selection-set reuse\n(best of 30, n=500)", reuse_res["sel500"], reuse_res["tst500"]))

    plots = generate_leakage_diagrams(summary, wrong_noise, right_noise, reuse_res["sel_sizes"], reuse_res["opt"])
    rep_path = save_leakage_report(audit_res, auc_res["top_features"], summary, reuse_res, d_prep)

    elapsed = round(time.perf_counter() - start_time, 2)
    print(f"\n[DONE] Data leakage pipeline completed in {elapsed}s.")
    print(f"       Diagnostic plots: {list(plots.values())}")
    print(f"       Report saved to:  {Path(rep_path).name}")

    return {
        "audit_res": audit_res,
        "summary": summary,
        "plots": plots,
        "report_path": rep_path
    }


if __name__ == "__main__":
    run_data_leakage_pipeline()
