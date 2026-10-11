"""
Course: Machine Learning (25SC2107E) - Module 4: Unsupervised Learning
Session: Anomaly Detection (Isolation Forest, One-Class SVM & Autoencoders)
Reference: M4-Session6-Anomaly_Detection-new.pptx

Topics covered:
1. Isolation Forest:
   - Recursive partitioning & path length h(x)
   - c(n) average path length derivation & anomaly score s(x, n) = 2^(-E[h(x)]/c(n))
   - Slide 6 worked numerical example verification on X = {1, 2, 3, 4, 5, 50}
   - Contamination parameter & threshold calibration
2. One-Class SVM:
   - Maximum-margin boundary enclosing normal region in RBF kernel space
   - Role of nu parameter (margin error bound)
   - Slide 12 worked numerical example verification (SV1=(2,2), SV2=(3,3), gamma=0.5, rho=0.35)
3. Autoencoder-based Anomaly Detection:
   - Bottleneck reconstruction error A(x) = ||x - x_hat||^2 as anomaly score
   - Slide 16 worked numerical example verification (Linear AE, threshold tau=0.5)
   - PCA/Linear reconstruction model on full feature space
4. Placement Prediction Dataset Application:
   - Standardized numeric features with zero target leakage
   - Benchmark evaluation against ground-truth 'IsAnomaly' column (Precision, Recall, F1, ROC-AUC)
   - Concordance / Jaccard similarity across detection algorithms
   - Multivariate anomaly profiling (identifying which features trigger anomalies)
5. Diagnostic visualization exports to Output/plots/.
6. Diagnostic text report generation to Output/Report/anomaly_report.txt.
"""

from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

import config
from src.data_utils import load_cleaned

ANOMALY_FEATURES = [
    "CGPA", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore",
    "AttendancePercent", "SoftSkillsRating", "Internships", "Projects",
    "Workshops", "Certifications", "Publications"
]
TARGET_ANOMALY = "IsAnomaly"
TARGET_CLASSIFICATION = "PlacementStatus"


# =====================================================================
# 1. Mathematical Formulations & Worked Numerical Examples (Slides 4-17)
# =====================================================================

def harmonic_number(i):
    """Compute harmonic number H(i) = sum_{k=1}^i (1/k)."""
    if i <= 0:
        return 0.0
    return sum(1.0 / k for k in range(1, i + 1))


def avg_path_length_c(n):
    """Compute c(n) = 2*H(n-1) - 2*(n-1)/n (Slide 5 & 6)."""
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.0
    return 2.0 * harmonic_number(n - 1) - (2.0 * (n - 1) / n)


def isolation_score(h, n):
    """Compute Isolation Forest anomaly score s(x, n) = 2^(-h / c(n)) (Slide 5 & 6)."""
    c = avg_path_length_c(n)
    return float(2.0 ** (-h / c))


def run_isolation_forest_worked_example():
    """Reproduce Slide 6 worked numerical example on n=6: X = {1, 2, 3, 4, 5, 50}."""
    n = 6
    c_val = avg_path_length_c(n)  # 2.90
    
    # Path lengths from slide 6
    points = [50, 1, 4]  # representative groups: 50, {1,2,3}, {4,5}
    path_lengths = {50: 1, 1: 3, 4: 4}
    
    results = {}
    for pt in [50, 1, 4]:
        h = path_lengths[pt]
        ratio = -h / c_val
        s = 2.0 ** ratio
        verdict = "Strong anomaly (s -> 1)" if s > 0.6 else "Normal"
        results[pt] = {
            "path_length": h,
            "neg_ratio": round(ratio, 3),
            "score": round(s, 3),
            "verdict": verdict
        }
        
    return {
        "n": n,
        "c_val": round(c_val, 2),
        "cases": results
    }


def rbf_kernel(x, y, gamma=0.5):
    """Compute RBF Gaussian kernel K(x, y) = exp(-gamma * ||x - y||^2) (Slide 11 & 12)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    sq_dist = np.sum((x - y) ** 2)
    return float(np.exp(-gamma * sq_dist))


def run_one_class_svm_worked_example():
    """Reproduce Slide 12 worked numerical example: SV1=(2,2), SV2=(3,3), gamma=0.5, rho=0.35."""
    sv1 = np.array([2.0, 2.0])
    sv2 = np.array([3.0, 3.0])
    alpha1 = 0.5
    alpha2 = 0.5
    gamma = 0.5
    rho = 0.35
    
    # Test A: x = (2.5, 2.5) - inside cluster
    xA = np.array([2.5, 2.5])
    k1_A = rbf_kernel(sv1, xA, gamma)
    k2_A = rbf_kernel(sv2, xA, gamma)
    f_A = alpha1 * k1_A + alpha2 * k2_A - rho
    
    # Test B: x = (10, 10) - far outlier
    xB = np.array([10.0, 10.0])
    k1_B = rbf_kernel(sv1, xB, gamma)
    k2_B = rbf_kernel(sv2, xB, gamma)
    f_B = alpha1 * k1_B + alpha2 * k2_B - rho
    
    return {
        "gamma": gamma,
        "rho": rho,
        "Test_A": {
            "x": [2.5, 2.5],
            "k1": round(k1_A, 4),
            "k2": round(k2_A, 4),
            "decision_val": round(f_A, 3),
            "verdict": "NORMAL (f(x) > 0)" if f_A > 0 else "ANOMALY"
        },
        "Test_B": {
            "x": [10.0, 10.0],
            "k1": round(k1_B, 6),
            "k2": round(k2_B, 6),
            "decision_val": round(f_B, 3),
            "verdict": "ANOMALY (f(x) < 0)" if f_B < 0 else "NORMAL"
        }
    }


def run_autoencoder_worked_example():
    """Reproduce Slide 16 worked numerical example: Linear AE, threshold tau=0.5."""
    tau = 0.5
    
    def encode(x, w=(0.5, 0.5)):
        return w[0] * x[0] + w[1] * x[1]
        
    def decode(z):
        return [z, z]
        
    def reconstruction_error(x, x_hat):
        return (x[0] - x_hat[0]) ** 2 + (x[1] - x_hat[1]) ** 2
        
    cases = {
        "Case 1 (normal)": [4.0, 4.0],
        "Case 2 (near-normal)": [5.0, 4.8],
        "Case 3 (anomaly)": [1.0, 9.0]
    }
    
    eval_results = {}
    for name, x in cases.items():
        z = encode(x)
        x_hat = decode(z)
        err = reconstruction_error(x, x_hat)
        verdict = "ANOMALY (>> tau)" if err > tau else "NORMAL"
        eval_results[name] = {
            "x": x,
            "z": round(z, 2),
            "x_hat": [round(v, 2) for v in x_hat],
            "error": round(err, 2),
            "verdict": verdict
        }
        
    return {
        "tau": tau,
        "cases": eval_results
    }


# =====================================================================
# 2. Placement Dataset Ingestion & Feature Preparation
# =====================================================================

def get_anomaly_data(df=None):
    """Load cleaned Placement dataset and extract standardized numeric features.
    
    Zero data leakage: Excludes targets (PlacementStatus, Salary Package, StudentID).
    Preserves ground-truth 'IsAnomaly' solely for benchmark evaluation.
    """
    data = load_cleaned() if df is None else df.copy()
    available_features = [f for f in ANOMALY_FEATURES if f in data.columns]
    
    X_raw = data[available_features].copy()
    for col in available_features:
        X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce")
        
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X_raw), columns=available_features, index=data.index
    )
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_imputed), columns=available_features, index=data.index
    )
    
    y_true = data[TARGET_ANOMALY].values if TARGET_ANOMALY in data.columns else None
    
    return data, available_features, X_scaled, y_true, scaler


# =====================================================================
# 3. Model Training & Anomaly Scoring Pipelines
# =====================================================================

def train_isolation_forest(X_scaled, contamination=0.035, random_state=42):
    """Fit Isolation Forest on Placement dataset."""
    t0 = time.perf_counter()
    iso = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    iso.fit(X_scaled)
    elapsed = round(time.perf_counter() - t0, 2)
    
    # decision_function gives negative of anomaly score: higher = more normal
    raw_scores = iso.decision_function(X_scaled)
    # Convert to anomaly score where higher = more anomalous
    anomaly_scores = -raw_scores
    preds = iso.predict(X_scaled)  # -1 = anomaly, 1 = normal
    binary_preds = (preds == -1).astype(int)
    
    return {
        "model": iso,
        "scores": anomaly_scores,
        "preds": binary_preds,
        "fit_time": elapsed,
        "contamination": contamination
    }


def train_one_class_svm(X_scaled, nu=0.035, sample_size=5000, random_state=42):
    """Fit One-Class SVM on a representative subsample (managing O(n^2)-O(n^3) complexity)."""
    t0 = time.perf_counter()
    n_samples = len(X_scaled)
    
    if n_samples > sample_size:
        rng = np.random.RandomState(random_state)
        sample_idx = rng.choice(n_samples, size=sample_size, replace=False)
        X_train_sub = X_scaled.iloc[sample_idx] if hasattr(X_scaled, "iloc") else X_scaled[sample_idx]
    else:
        X_train_sub = X_scaled
        
    oc_svm = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
    oc_svm.fit(X_train_sub)
    fit_time = round(time.perf_counter() - t0, 2)
    
    # Score all samples
    t1 = time.perf_counter()
    decision_vals = oc_svm.decision_function(X_scaled)
    preds = oc_svm.predict(X_scaled)  # -1 = anomaly, 1 = normal
    score_time = round(time.perf_counter() - t1, 2)
    
    anomaly_scores = -decision_vals  # Higher = more anomalous
    binary_preds = (preds == -1).astype(int)
    
    return {
        "model": oc_svm,
        "scores": anomaly_scores,
        "preds": binary_preds,
        "fit_time": fit_time,
        "score_time": score_time,
        "nu": nu,
        "train_samples": len(X_train_sub),
        "n_support_vectors": len(oc_svm.support_)
    }


def train_autoencoder_reconstruction(X_scaled, n_components=3, quantile_threshold=0.965):
    """Linear Autoencoder / Subspace Reconstruction Model using PCA bottleneck (Slide 15 & 17)."""
    t0 = time.perf_counter()
    # A linear autoencoder with MSE loss learns the identical subspace as PCA (Slide 17)
    pca = PCA(n_components=n_components, random_state=42)
    Z = pca.fit_transform(X_scaled)
    X_reconstructed = pca.inverse_transform(Z)
    fit_time = round(time.perf_counter() - t0, 2)
    
    # Reconstruction error A(x) = ||x - x_hat||^2
    errors = np.sum((X_scaled.values - X_reconstructed) ** 2, axis=1)
    threshold = float(np.quantile(errors, quantile_threshold))
    binary_preds = (errors > threshold).astype(int)
    
    return {
        "model": pca,
        "scores": errors,
        "threshold": round(threshold, 3),
        "preds": binary_preds,
        "fit_time": fit_time,
        "explained_variance_ratio": round(float(np.sum(pca.explained_variance_ratio_)), 4)
    }


# =====================================================================
# 4. Evaluation & Concordance Benchmarks
# =====================================================================

def evaluate_anomaly_methods(y_true, iso_res, ocsvm_res, ae_res):
    """Compute quantitative classification and ranking metrics against ground-truth IsAnomaly."""
    methods = {
        "Isolation Forest": iso_res,
        "One-Class SVM": ocsvm_res,
        "Autoencoder (Reconstruction)": ae_res
    }
    
    benchmark_table = []
    roc_data = {}
    pr_data = {}
    
    for name, res in methods.items():
        preds = res["preds"]
        scores = res["scores"]
        
        prec = float(precision_score(y_true, preds, zero_division=0))
        rec = float(recall_score(y_true, preds, zero_division=0))
        f1 = float(f1_score(y_true, preds, zero_division=0))
        auc = float(roc_auc_score(y_true, scores))
        cm = confusion_matrix(y_true, preds)
        
        fpr, tpr, _ = roc_curve(y_true, scores)
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, scores)
        
        roc_data[name] = (fpr, tpr, auc)
        pr_data[name] = (rec_curve, prec_curve)
        
        benchmark_table.append({
            "Method": name,
            "Precision": round(prec * 100, 2),
            "Recall": round(rec * 100, 2),
            "F1-Score": round(f1 * 100, 2),
            "ROC-AUC": round(auc, 4),
            "True Positives": int(cm[1, 1]),
            "False Positives": int(cm[0, 1]),
            "False Negatives": int(cm[1, 0]),
            "True Negatives": int(cm[0, 0]),
            "Fit Time (s)": res["fit_time"]
        })
        
    bench_df = pd.DataFrame(benchmark_table)
    
    # Pairwise Jaccard concordance
    p_iso = iso_res["preds"]
    p_svm = ocsvm_res["preds"]
    p_ae = ae_res["preds"]
    
    def jaccard(a, b):
        intersection = np.sum((a == 1) & (b == 1))
        union = np.sum((a == 1) | (b == 1))
        return round(float(intersection / union) * 100, 2) if union > 0 else 0.0
        
    concordance = {
        "IsoForest vs OCSVM": jaccard(p_iso, p_svm),
        "IsoForest vs Autoencoder": jaccard(p_iso, p_ae),
        "OCSVM vs Autoencoder": jaccard(p_svm, p_ae)
    }
    
    return bench_df, roc_data, pr_data, concordance


# =====================================================================
# 5. Diagnostic Visualization Exports (Output/plots/)
# =====================================================================

def generate_anomaly_diagrams(worked_iso, worked_svm, worked_ae, df, bench_df, roc_data, pr_data, iso_res, X_scaled):
    """Generate and save publication-quality diagnostic plots for Anomaly Detection."""
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    generated_plots = {}
    
    # -------------------------------------------------------------
    # Plot 1: Visual Verification of 3 Worked Numerical Examples (Slides 6, 12, 16)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel A: Isolation Forest Path Lengths & Scores
    pts = [50, 1, 4]
    scores_iso = [worked_iso["cases"][p]["score"] for p in pts]
    h_lens = [worked_iso["cases"][p]["path_length"] for p in pts]
    pt_names = ["Point 50\n(h=1, outlier)", "Points 1..3\n(h=3, normal)", "Points 4..5\n(h=4, dense)"]
    colors_iso = ["#e74c3c", "#3498db", "#2ecc71"]
    
    bars = axes[0].bar(pt_names, scores_iso, color=colors_iso, width=0.55, edgecolor="black", alpha=0.85)
    axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=1.2, label="Neutral s=0.5")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Anomaly Score s(x, n) = 2^(-h/c)", fontweight="bold")
    axes[0].set_title(f"1. Isolation Forest (Slide 6)\nc(6)={worked_iso['c_val']:.2f} | s(50)={worked_iso['cases'][50]['score']}", fontsize=11, fontweight="bold")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    for bar, sc in zip(bars, scores_iso):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"s={sc}", ha="center", fontweight="bold")
        
    # Panel B: One-Class SVM Decision Values
    svm_labels = ["Test A: (2.5, 2.5)\n(Inside cluster)", "Test B: (10, 10)\n(Far outlier)"]
    svm_f_vals = [worked_svm["Test_A"]["decision_val"], worked_svm["Test_B"]["decision_val"]]
    svm_colors = ["#2ecc71", "#e74c3c"]
    bars_svm = axes[1].bar(svm_labels, svm_f_vals, color=svm_colors, width=0.45, edgecolor="black", alpha=0.85)
    axes[1].axhline(0.0, color="black", linestyle="-", linewidth=1.2, label="Boundary f(x)=0")
    axes[1].set_ylabel("Signed Distance f(x)", fontweight="bold")
    axes[1].set_title(f"2. One-Class SVM (Slide 12)\nrho={worked_svm['rho']}, gamma={worked_svm['gamma']}", fontsize=11, fontweight="bold")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle=":", alpha=0.5)
    for bar, val in zip(bars_svm, svm_f_vals):
        y_pos = bar.get_height() + 0.03 if val >= 0 else bar.get_height() - 0.08
        axes[1].text(bar.get_x() + bar.get_width() / 2, y_pos, f"f={val:+0.3f}", ha="center", fontweight="bold")
        
    # Panel C: Autoencoder Reconstruction Errors
    ae_labels = ["Case 1: (4,4)\nNormal", "Case 2: (5,4.8)\nNear-normal", "Case 3: (1,9)\nRelationship Broken"]
    ae_errors = [worked_ae["cases"]["Case 1 (normal)"]["error"],
                 worked_ae["cases"]["Case 2 (near-normal)"]["error"],
                 worked_ae["cases"]["Case 3 (anomaly)"]["error"]]
    ae_colors = ["#2ecc71", "#3498db", "#e74c3c"]
    bars_ae = axes[2].bar(ae_labels, ae_errors, color=ae_colors, width=0.5, edgecolor="black", alpha=0.85)
    axes[2].axhline(worked_ae["tau"], color="red", linestyle="--", linewidth=1.2, label=f"Threshold tau={worked_ae['tau']}")
    axes[2].set_ylabel("Reconstruction Error ||x - x_hat||^2", fontweight="bold")
    axes[2].set_title(f"3. Autoencoder Preview (Slide 16)\nPattern x1~x2 broken in Case 3", fontsize=11, fontweight="bold")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, linestyle=":", alpha=0.5)
    for bar, err in zip(bars_ae, ae_errors):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, f"Err={err}", ha="center", fontweight="bold")
        
    fig.suptitle("Verification of M4 Session 6 Hand-Worked Numerical Examples", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plot_worked_path = config.PLOTS_DIR / "anomaly_worked_examples.png"
    plt.savefig(plot_worked_path, dpi=150, bbox_inches="tight")
    plt.close()
    generated_plots["worked_examples"] = plot_worked_path.name
    
    # -------------------------------------------------------------
    # Plot 2: Isolation Forest Score Distribution & Contamination Threshold (Slide 7)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    iso_scores = iso_res["scores"]
    y_true = df[TARGET_ANOMALY].values
    
    sns.histplot(iso_scores[y_true == 0], bins=60, color="#2ecc71", label="Ground Truth Normal (0)", kde=True, ax=ax, alpha=0.5)
    sns.histplot(iso_scores[y_true == 1], bins=40, color="#e74c3c", label="Ground Truth Anomaly (1)", kde=True, ax=ax, alpha=0.7)
    
    # Contamination threshold cutoff
    thresh_val = float(np.quantile(iso_scores, 1.0 - iso_res["contamination"]))
    ax.axvline(thresh_val, color="black", linestyle="--", linewidth=2.0, label=f"Contamination Cutoff (3.5% = {thresh_val:.3f})")
    
    ax.set_title("Isolation Forest Anomaly Score Distribution on 50,000 Placement Students", fontsize=12, fontweight="bold")
    ax.set_xlabel("Anomaly Score (Higher = More Anomalous)")
    ax.set_ylabel("Student Count")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plot_dist_path = config.PLOTS_DIR / "anomaly_score_distribution.png"
    plt.savefig(plot_dist_path, dpi=150, bbox_inches="tight")
    plt.close()
    generated_plots["score_distribution"] = plot_dist_path.name
    
    # -------------------------------------------------------------
    # Plot 3: Comparative ROC & Precision-Recall Curves (Slide 18)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    colors_map = {"Isolation Forest": "#e74c3c", "One-Class SVM": "#3498db", "Autoencoder (Reconstruction)": "#9b59b6"}
    
    # ROC Curves
    for name, (fpr, tpr, auc_val) in roc_data.items():
        axes[0].plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", color=colors_map[name], linewidth=2.0)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (0.50)")
    axes[0].set_title("Receiver Operating Characteristic (ROC)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # Precision-Recall Curves
    for name, (rec_c, prec_c) in pr_data.items():
        axes[1].plot(rec_c, prec_c, label=f"{name}", color=colors_map[name], linewidth=2.0)
    base_rate = float(np.mean(y_true))
    axes[1].axhline(base_rate, color="k", linestyle="--", alpha=0.5, label=f"Base Rate ({base_rate*100:.1f}%)")
    axes[1].set_title("Precision-Recall Curve (PR)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    fig.suptitle("Unsupervised Anomaly Detection Benchmark vs Ground-Truth IsAnomaly", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plot_roc_path = config.PLOTS_DIR / "anomaly_roc_pr_curves.png"
    plt.savefig(plot_roc_path, dpi=150, bbox_inches="tight")
    plt.close()
    generated_plots["roc_pr_curves"] = plot_roc_path.name
    
    # -------------------------------------------------------------
    # Plot 4: Anomaly Profile Disruption (Why points are flagged)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    pred_iso = iso_res["preds"]
    
    key_features = [
        ("CGPA", "AptitudeTestScore"),
        ("CodingTestScore", "AttendancePercent"),
        ("CGPA", "CodingTestScore"),
        ("MockInterviewScore", "SoftSkillsRating")
    ]
    
    for idx, (f1_name, f2_name) in enumerate(key_features):
        ax = axes[idx // 2, idx % 2]
        if f1_name in df.columns and f2_name in df.columns:
            # Background normal
            normal_mask = (pred_iso == 0)
            ax.scatter(df.loc[normal_mask, f1_name], df.loc[normal_mask, f2_name],
                       c="#bdc3c7", label="Normal Students", alpha=0.25, s=8, edgecolors="none")
            # Foreground anomalies
            anom_mask = (pred_iso == 1)
            ax.scatter(df.loc[anom_mask, f1_name], df.loc[anom_mask, f2_name],
                       c="#e74c3c", label=f"Detected Anomalies (N={int(np.sum(anom_mask))})", alpha=0.75, s=20, edgecolors="black", linewidths=0.3)
            ax.set_title(f"{f1_name} vs {f2_name}", fontsize=11, fontweight="bold")
            ax.set_xlabel(f1_name)
            ax.set_ylabel(f2_name)
            ax.grid(True, linestyle=":", alpha=0.5)
            if idx == 0:
                ax.legend(loc="upper left")
                
    fig.suptitle("Multivariate Feature Profile Violations Identified by Isolation Forest", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plot_profile_path = config.PLOTS_DIR / "anomaly_detected_profiles.png"
    plt.savefig(plot_profile_path, dpi=150, bbox_inches="tight")
    plt.close()
    generated_plots["detected_profiles"] = plot_profile_path.name
    
    return generated_plots


# =====================================================================
# 6. Diagnostic Report Generation (Output/Report/anomaly_report.txt)
# =====================================================================

def save_anomaly_report(worked_iso, worked_svm, worked_ae, bench_df, concordance, n_total, n_anomalies):
    """Save comprehensive Anomaly Detection diagnostic report to config.REPORTS_DIR."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = config.REPORTS_DIR / "anomaly_report.txt"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("MODULE 4 · SESSION 6: ANOMALY DETECTION REPORT\n")
        f.write("Course: Machine Learning (25SC2107E) | Placement Prediction Dataset (50,000 Records)\n")
        f.write("Reference: M4-Session6-Anomaly_Detection-new.pptx\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. WORKED NUMERICAL EXAMPLES VERIFICATION (SLIDES 6, 12, 16):\n")
        f.write("   A. Isolation Forest (Slide 6):\n")
        f.write(f"      Dataset: n={worked_iso['n']}, c({worked_iso['n']}) = {worked_iso['c_val']:.2f}\n")
        for pt, data in worked_iso["cases"].items():
            f.write(f"      - Point {pt:<2}: path h={data['path_length']}, ratio={data['neg_ratio']:<6}, score s={data['score']:.3f} -> {data['verdict']}\n")
            
        f.write("\n   B. One-Class SVM (Slide 12):\n")
        f.write(f"      Parameters: gamma={worked_svm['gamma']}, rho={worked_svm['rho']}, SV1=(2,2), SV2=(3,3)\n")
        f.write(f"      - Test A (2.5, 2.5): f(x) = {worked_svm['Test_A']['decision_val']:+0.3f} -> {worked_svm['Test_A']['verdict']}\n")
        f.write(f"      - Test B (10,  10) : f(x) = {worked_svm['Test_B']['decision_val']:+0.3f} -> {worked_svm['Test_B']['verdict']}\n")
        
        f.write("\n   C. Autoencoder Preview (Slide 16):\n")
        f.write(f"      Model: z = 0.5*x1 + 0.5*x2, x_hat = [z, z], threshold tau = {worked_ae['tau']}\n")
        for cname, cdata in worked_ae["cases"].items():
            f.write(f"      - {cname:<22}: x={cdata['x']} -> z={cdata['z']} -> x_hat={cdata['x_hat']} -> Error={cdata['error']:.2f} ({cdata['verdict']})\n")
            
        f.write("\n2. PLACEMENT PREDICTION DATASET BENCHMARK EVALUATION:\n")
        f.write(f"   Total student records: {n_total:,}\n")
        f.write(f"   Ground truth anomalies (IsAnomaly=1): {n_anomalies:,} ({n_anomalies/n_total*100:.2f}% contamination)\n\n")
        f.write(bench_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("3. ALGORITHM CONCORDANCE (JACCARD OVERLAP OF FLAGGED OUTLIERS):\n")
        for pair_name, jacc in concordance.items():
            f.write(f"   - {pair_name:<30}: {jacc:.2f}% overlap\n")
            
        f.write("\n4. COMPARATIVE ARCHITECTURAL SYNTHESIS (SLIDE 18):\n")
        f.write("   " + "-" * 75 + "\n")
        f.write(f"   {'Criterion':<22} | {'Isolation Forest':<16} | {'One-Class SVM':<16} | {'Autoencoder':<16}\n")
        f.write("   " + "-" * 75 + "\n")
        f.write(f"   {'Anomaly Signal':<22} | {'Short path length':<16} | {'Outside boundary':<16} | {'High recon error':<16}\n")
        f.write(f"   {'Key Parameter':<22} | {'contamination':<16} | {'nu (bound)':<16} | {'bottleneck & tau':<16}\n")
        f.write(f"   {'Large n Scaling':<22} | {'Yes (near-linear)':<16} | {'Costly O(n^2-n^3)':<16} | {'Yes (mini-batch)':<16}\n")
        f.write(f"   {'Non-Linear Bound':<22} | {'Yes (tree splits)':<16} | {'Yes (RBF kernel)':<16} | {'Yes (activations)':<16}\n")
        f.write(f"   {'Typical Domain':<22} | {'Tabular, fraud':<16} | {'Moderate tabular':<16} | {'Images, sequences':<16}\n")
        f.write("   " + "-" * 75 + "\n\n")
        
        f.write("5. CONTRAST WITH LINEAR REGRESSION (SLIDES 9, 14, 17):\n")
        f.write("   - Linear regression is supervised and harmed by outliers (distorting beta coefficients).\n")
        f.write("   - Unsupervised anomaly detectors are built specifically to isolate and expose anomalies.\n")
        f.write("   - Contamination and nu explicitly govern false alarm bounds without modifying underlying data.\n")
    return report_path


# =====================================================================
# 7. Pipeline Runner
# =====================================================================

def run_anomaly_pipeline(df=None, random_state=42):
    """Execute complete Anomaly Detection pipeline standalone."""
    print("=" * 70)
    print("MODULE 4 · SESSION 6: ANOMALY DETECTION PIPELINE")
    print("=" * 70)
    start_time = time.perf_counter()
    
    print("[1/5] Verifying M4 Session 6 hand-worked numerical examples...")
    worked_iso = run_isolation_forest_worked_example()
    worked_svm = run_one_class_svm_worked_example()
    worked_ae = run_autoencoder_worked_example()
    print(f"      Isolation Forest : s(50)={worked_iso['cases'][50]['score']}, s(1)={worked_iso['cases'][1]['score']}, s(4)={worked_iso['cases'][4]['score']}")
    print(f"      One-Class SVM    : f(2.5,2.5)={worked_svm['Test_A']['decision_val']}, f(10,10)={worked_svm['Test_B']['decision_val']}")
    print(f"      Autoencoder      : Err(4,4)={worked_ae['cases']['Case 1 (normal)']['error']}, Err(1,9)={worked_ae['cases']['Case 3 (anomaly)']['error']}")
    
    print("[2/5] Loading 50,000 Placement records & standardizing numeric metrics...")
    data, features, X_scaled, y_true, scaler = get_anomaly_data(df=df)
    n_anom = int(np.sum(y_true)) if y_true is not None else 0
    contamination = (n_anom / len(data)) if n_anom > 0 else 0.035
    print(f"      Features: {len(features)} | Ground-truth anomalies: {n_anom:,} ({contamination*100:.2f}%)")
    
    print(f"[3/5] Fitting Isolation Forest (contamination={contamination:.3f})...")
    iso_res = train_isolation_forest(X_scaled, contamination=contamination, random_state=random_state)
    print(f"      Isolation Forest fit in {iso_res['fit_time']}s | Flagged: {int(np.sum(iso_res['preds'])):,} candidates")
    
    print(f"[4/5] Fitting One-Class SVM (nu={contamination:.3f}) and Subspace Autoencoder...")
    ocsvm_res = train_one_class_svm(X_scaled, nu=contamination, sample_size=5000, random_state=random_state)
    print(f"      One-Class SVM fit in {ocsvm_res['fit_time']}s (score {ocsvm_res['score_time']}s) | Support vectors: {ocsvm_res['n_support_vectors']}")
    
    ae_res = train_autoencoder_reconstruction(X_scaled, n_components=3, quantile_threshold=1.0 - contamination)
    print(f"      Autoencoder fit in {ae_res['fit_time']}s | Reconstructed variance: {ae_res['explained_variance_ratio']*100:.1f}%")
    
    print("[5/5] Evaluating against ground-truth IsAnomaly & generating diagnostic plots...")
    bench_df, roc_data, pr_data, concordance = evaluate_anomaly_methods(y_true, iso_res, ocsvm_res, ae_res)
    print(bench_df[["Method", "Precision", "Recall", "F1-Score", "ROC-AUC"]])
    
    plots = generate_anomaly_diagrams(worked_iso, worked_svm, worked_ae, data, bench_df, roc_data, pr_data, iso_res, X_scaled)
    rep_path = save_anomaly_report(worked_iso, worked_svm, worked_ae, bench_df, concordance, len(data), n_anom)
    
    elapsed = round(time.perf_counter() - start_time, 2)
    print(f"\n[DONE] Anomaly Detection pipeline completed successfully in {elapsed}s.")
    print(f"       Diagnostic plots: {list(plots.values())}")
    print(f"       Report: {Path(rep_path).name}")
    
    return {
        "worked_iso": worked_iso,
        "worked_svm": worked_svm,
        "worked_ae": worked_ae,
        "bench_df": bench_df,
        "concordance": concordance,
        "plots": plots,
        "report_path": rep_path
    }


if __name__ == "__main__":
    run_anomaly_pipeline()
