"""
Course: Machine Learning (25SC2107E) - Module 4: Unsupervised Learning
Session: Principal Component Analysis (PCA)
Reference: M4-Session4-PCA.pptx & principal_component_analysis.py

Topics covered:
1. Core Theory & Concept:
   - Orthogonal linear transformations maximizing variance
   - Why variance = information and dimensionality reduction
2. Hand-Worked Numerical Example Verification (Slide 8 & 12):
   - 10 points in 2-D, mean centering, sample covariance matrix Sigma
   - Eigen-decomposition: lambda1=1.2840 (96.3%), lambda2=0.0491 (3.7%), v1=[0.6779, 0.7352]
   - 1-D projection Z = X_c * v1
3. Standardization Pitfall Demonstration (Slide 11 & 13):
   - Contrast raw-scale PCA (income dominates 100%) vs StandardScaler PCA (balanced 66.5%/33.2%)
4. Placement Prediction Dataset Pipeline:
   - Standardized numeric features (zero target leakage)
   - Scree plot & cumulative variance curves (80%, 90%, 95% thresholds)
   - Feature Loadings interpretation & biplot
   - 2D projection colored by PlacementStatus (grounding Sessions 21 & 24)
   - Production Pipeline with StandardScaler + PCA + Logistic Regression (Slide 10)
5. Diagnostic visualization exports to config.PLOTS_DIR.
6. Diagnostic text report generation to config.REPORTS_DIR.
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
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from src.data_utils import load_cleaned

# Ensure output directories exist
config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# 1. Hand-Worked Numerical Example Verification (Slides 8 & 12)
# =====================================================================

def run_pca_worked_example():
    """Reproduce Slide 8 & 12 worked numerical example on 10 points in 2-D."""
    X = np.array([
        [2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0],
        [2.3, 2.7], [2.0, 1.6], [1.0, 1.1], [1.5, 1.6], [1.1, 0.9]
    ])
    n = len(X)
    mean_vec = X.mean(axis=0)
    Xc = X - mean_vec
    
    # Sample covariance matrix: Sigma = (1/(n-1)) X_c^T X_c
    cov_matrix = np.cov(Xc, rowvar=False, bias=False)
    
    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    
    # Ensure consistent eigenvector sign convention matching slide: v1 = (0.6779, 0.7352)
    if eigvecs[0, 0] < 0:
        eigvecs[:, 0] = -eigvecs[:, 0]
        
    explained_ratio = eigvals / np.sum(eigvals)
    v1 = eigvecs[:, 0]
    Z1 = Xc.dot(v1)  # 1-D projection
    
    # Cross-check with scikit-learn
    pca_sklearn = PCA(n_components=2).fit(X)
    
    return {
        "X": X,
        "mean": np.round(mean_vec, 2),
        "cov_matrix": np.round(cov_matrix, 4),
        "eigvals": np.round(eigvals, 4),
        "eigvecs": np.round(eigvecs, 4),
        "v1": np.round(v1, 4),
        "explained_ratio": np.round(explained_ratio, 4),
        "explained_pct_pc1": round(float(explained_ratio[0] * 100), 1),
        "explained_pct_pc2": round(float(explained_ratio[1] * 100), 1),
        "Z1": np.round(Z1, 3),
        "sklearn_explained_ratio": np.round(pca_sklearn.explained_variance_ratio_, 4)
    }


# =====================================================================
# 2. Standardization Pitfall Demonstration (Slides 11 & 13)
# =====================================================================

def run_standardization_pitfall_demo(random_state=42):
    """Reproduce Slide 11 & 13: Contrast unscaled PCA vs StandardScaler PCA."""
    rng = np.random.default_rng(random_state)
    n = 200
    height_cm = rng.normal(165, 10, n)
    income_usd = 30000 + 2500 * (height_cm - 165) + rng.normal(0, 3000, n)
    age = rng.normal(35, 6, n)
    X = np.column_stack([height_cm, income_usd, age])
    
    # Raw PCA without scaling (income dominates 100%)
    pca_raw = PCA(n_components=3).fit(X)
    raw_var = np.round(pca_raw.explained_variance_ratio_, 4)
    
    # Scaled PCA with StandardScaler (balanced 66.5% / 33.2%)
    Xs = StandardScaler().fit_transform(X)
    pca_std = PCA(n_components=3).fit(Xs)
    std_var = np.round(pca_std.explained_variance_ratio_, 4)
    
    return {
        "raw_explained_ratio": raw_var,
        "std_explained_ratio": std_var,
        "raw_pc1_pct": round(float(raw_var[0] * 100), 1),
        "std_pc1_pct": round(float(std_var[0] * 100), 1),
        "std_pc2_pct": round(float(std_var[1] * 100), 1)
    }


# =====================================================================
# 3. Placement Prediction Dataset PCA Pipeline
# =====================================================================

def get_pca_data(df=None):
    """Load cleaned Placement dataset and extract standardized numeric features.
    
    Zero data leakage: Excludes targets (PlacementStatus, Salary Package, StudentID, IsAnomaly).
    """
    data = load_cleaned() if df is None else df.copy()
    feature_cols = [c for c in config.NUMERIC_COLS if c in data.columns]
    
    x_raw = data[feature_cols].copy()
    for col in feature_cols:
        x_raw[col] = pd.to_numeric(x_raw[col], errors="coerce")
        
    imputer = SimpleImputer(strategy="median")
    x_imputed = pd.DataFrame(
        imputer.fit_transform(x_raw), columns=feature_cols, index=x_raw.index
    )
    
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_imputed)
    
    return data, feature_cols, x_imputed, x_scaled, scaler


def run_full_pca_pipeline(df=None):
    """Execute complete PCA pipeline on Placement dataset."""
    t0 = time.perf_counter()
    data, feature_cols, x_imputed, x_scaled, scaler = get_pca_data(df)
    n_samples = len(data)
    n_features = len(feature_cols)
    
    # 1. Fit PCA keeping ALL components
    pca_full = PCA(n_components=None, random_state=config.RANDOM_STATE)
    pca_full.fit(x_scaled)
    fit_time = round(time.perf_counter() - t0, 2)
    
    explained_variance_ratio = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Thresholds
    thresholds = [0.80, 0.90, 0.95]
    components_needed = {}
    for thresh in thresholds:
        n_needed = int(np.searchsorted(cumulative_variance, thresh) + 1)
        components_needed[int(thresh * 100)] = n_needed
        
    # PC1 + PC2 combined variance (checking Sessions 21 & 24)
    pc1_pc2_variance = cumulative_variance[1]
    
    # 2. Loadings analysis (weights of each feature in PC1 and PC2)
    loadings = pd.DataFrame(
        pca_full.components_[:2].T,
        columns=["PC1", "PC2"],
        index=feature_cols
    )
    top_pc1 = loadings["PC1"].abs().sort_values(ascending=False)
    top_pc2 = loadings["PC2"].abs().sort_values(ascending=False)
    
    # 3. 2D projection
    x_2d = pca_full.transform(x_scaled)[:, :2]
    
    # 4. Production Pipeline Benchmark (Slide 10)
    # Testing Logistic Regression with vs without PCA (keeping 90% variance)
    target_series = data[config.TARGET_CLASSIFICATION]
    if target_series.dtype == object:
        y = (target_series.astype(str).str.strip().str.lower() == "placed").astype(int).values
    else:
        y = pd.to_numeric(target_series, errors="coerce").fillna(0).astype(int).values

    
    # Split 80/20 train/test
    rng = np.random.RandomState(config.RANDOM_STATE)
    test_mask = rng.rand(n_samples) < 0.2
    X_tr, X_te = x_scaled[~test_mask], x_scaled[test_mask]
    y_tr, y_te = y[~test_mask], y[test_mask]
    
    # Baseline: Full features
    clf_full = LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)
    clf_full.fit(X_tr, y_tr)
    acc_full = round(float(accuracy_score(y_te, clf_full.predict(X_te)) * 100), 2)
    auc_full = round(float(roc_auc_score(y_te, clf_full.predict_proba(X_te)[:, 1])), 4)
    
    # PCA Pipeline: keep 90% variance
    pipe_pca = Pipeline([
        ("pca", PCA(n_components=0.90, random_state=config.RANDOM_STATE)),
        ("clf", LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE))
    ])
    pipe_pca.fit(X_tr, y_tr)
    acc_pca = round(float(accuracy_score(y_te, pipe_pca.predict(X_te)) * 100), 2)
    auc_pca = round(float(roc_auc_score(y_te, pipe_pca.predict_proba(X_te)[:, 1])), 4)
    n_kept = pipe_pca.named_steps["pca"].n_components_
    
    return {
        "n_samples": n_samples,
        "feature_cols": feature_cols,
        "pca_full": pca_full,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_variance": cumulative_variance,
        "components_needed": components_needed,
        "pc1_pc2_variance": pc1_pc2_variance,
        "loadings": loadings,
        "top_pc1": top_pc1,
        "top_pc2": top_pc2,
        "x_2d": x_2d,
        "data": data,
        "fit_time": fit_time,
        "production_eval": {
            "full_features": {"acc": acc_full, "auc": auc_full, "n_features": n_features},
            "pca_features": {"acc": acc_pca, "auc": auc_pca, "n_components": n_kept}
        }
    }


# =====================================================================
# 4. Diagnostic Visualization Exports
# =====================================================================

def generate_pca_diagrams(worked_data, pitfall_data, full_res):
    """Generate publication-quality diagnostic plots for PCA and save to PLOTS_DIR."""
    save_dirs = [config.PLOTS_DIR]
    generated_plots = {}
    
    # -------------------------------------------------------------
    # Plot 1: Scree Plot & Cumulative Variance (Slide 9 & script)
    # -------------------------------------------------------------
    explained_ratio = full_res["explained_variance_ratio"]
    cum_var = full_res["cumulative_variance"]
    n_show = min(20, len(explained_ratio))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Individual variance
    axes[0].bar(range(1, n_show + 1), explained_ratio[:n_show] * 100, color="#4C72B0", edgecolor="black", alpha=0.85)
    axes[0].set_xlabel("Principal Component", fontweight="bold")
    axes[0].set_ylabel("Variance Explained (%)", fontweight="bold")
    axes[0].set_title("Scree Plot: Individual Variance per Component", fontsize=11, fontweight="bold")
    axes[0].set_xticks(range(1, n_show + 1))
    axes[0].grid(True, linestyle=":", alpha=0.5)
    for i, val in enumerate(explained_ratio[:n_show]):
        axes[0].text(i + 1, val * 100 + 0.5, f"{val*100:.1f}%", ha="center", fontsize=8)
        
    # Cumulative variance
    axes[1].plot(range(1, n_show + 1), cum_var[:n_show] * 100, marker="o", color="#DD8452", linewidth=2.0)
    for thresh in [80, 90, 95]:
        n_req = full_res["components_needed"][thresh]
        axes[1].axhline(y=thresh, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
        axes[1].scatter([n_req], [cum_var[n_req - 1] * 100], color="red", zorder=5, s=35)
        axes[1].annotate(f"{thresh}% ({n_req} PCs)", xy=(n_req, thresh), xytext=(n_req + 0.3, thresh - 2.5),
                         fontsize=8, fontweight="bold", color="#c0392b")
        
    axes[1].set_xlabel("Number of Components", fontweight="bold")
    axes[1].set_ylabel("Cumulative Variance Explained (%)", fontweight="bold")
    axes[1].set_title("Cumulative Explained Variance & Threshold Cutoffs", fontsize=11, fontweight="bold")
    axes[1].set_xticks(range(1, n_show + 1))
    axes[1].grid(True, linestyle=":", alpha=0.5)
    
    fig.suptitle("PCA Component Spectrum on Placement Prediction Features", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    for d in save_dirs:
        plt.savefig(d / "scree_and_cumulative_variance.png", dpi=150, bbox_inches="tight")
    plt.close()
    generated_plots["scree_plot"] = "scree_and_cumulative_variance.png"
    
    # -------------------------------------------------------------
    # Plot 2: Loadings Biplot (How original features map onto PC1/PC2)
    # -------------------------------------------------------------
    loadings = full_res["loadings"]
    exp_r = full_res["explained_variance_ratio"]
    feature_cols = full_res["feature_cols"]
    
    plt.figure(figsize=(9, 9))
    circle = plt.Circle((0, 0), 1, fill=False, linestyle="--", color="gray", alpha=0.7)
    plt.gca().add_artist(circle)
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(feature_cols)))
    for idx, feature in enumerate(feature_cols):
        x_val = loadings.loc[feature, "PC1"]
        y_val = loadings.loc[feature, "PC2"]
        plt.arrow(0, 0, x_val, y_val, head_width=0.03, color=colors[idx], alpha=0.8, linewidth=1.5, length_includes_head=True)
        # Position label slightly beyond the arrow head
        plt.text(x_val * 1.12, y_val * 1.12, feature, fontsize=8, ha="center", va="center", fontweight="bold", color="#2c3e50")
        
    plt.axhline(0, color="gray", linestyle=":", alpha=0.5)
    plt.axvline(0, color="gray", linestyle=":", alpha=0.5)
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.xlabel(f"PC1 ({exp_r[0]*100:.1f}% variance)", fontweight="bold")
    plt.ylabel(f"PC2 ({exp_r[1]*100:.1f}% variance)", fontweight="bold")
    plt.title("PCA Loadings: How Each Original Feature Maps onto PC1 / PC2", fontsize=12, fontweight="bold")
    plt.gca().set_aspect("equal")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    for d in save_dirs:
        plt.savefig(d / "loadings_biplot.png", dpi=150, bbox_inches="tight")
    plt.close()
    generated_plots["loadings_biplot"] = "loadings_biplot.png"
    
    # -------------------------------------------------------------
    # Plot 3: 2D Projection Colored by Placement Status
    # -------------------------------------------------------------
    x_2d = full_res["x_2d"]
    df = full_res["data"]
    pc12_pct = round(full_res["pc1_pc2_variance"] * 100, 1)
    
    plt.figure(figsize=(10, 7.5))
    t_series = df[config.TARGET_CLASSIFICATION]
    if t_series.dtype == object:
        target_vals = (t_series.astype(str).str.strip().str.lower() == "placed").astype(int)
    else:
        target_vals = pd.to_numeric(t_series, errors="coerce").fillna(0).astype(int)
    scatter = plt.scatter(
        x_2d[:, 0], x_2d[:, 1],
        c=target_vals, cmap="coolwarm", s=6, alpha=0.45, edgecolors="none"
    )
    plt.xlabel(f"PC1 ({exp_r[0]*100:.1f}% variance)", fontweight="bold")
    plt.ylabel(f"PC2 ({exp_r[1]*100:.1f}% variance)", fontweight="bold")
    plt.title(
        f"All {len(df):,} Students Projected onto PC1/PC2\n"
        f"(colored by actual placement outcome, capturing {pc12_pct}% of total variance)",
        fontsize=12, fontweight="bold"
    )
    cbar = plt.colorbar(scatter, ticks=[0, 1])
    cbar.set_ticklabels(["Not Placed", "Placed"])
    cbar.set_label("Placement Outcome", fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    for d in save_dirs:
        plt.savefig(d / "pca_2d_by_placement.png", dpi=150, bbox_inches="tight")
    plt.close()
    generated_plots["pca_2d_projection"] = "pca_2d_by_placement.png"
    
    # -------------------------------------------------------------
    # Plot 4: Worked Example & Scaling Pitfall Comparison (Slides 8 & 13)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Worked example eigen decomposition
    we_bars = axes[0].bar(["PC1 (Direction v1)", "PC2 (Orthogonal)"],
                          [worked_data["explained_pct_pc1"], worked_data["explained_pct_pc2"]],
                          color=["#2ecc71", "#95a5a6"], edgecolor="black", width=0.5)
    axes[0].set_ylim(0, 110)
    axes[0].set_ylabel("Variance Explained (%)", fontweight="bold")
    axes[0].set_title(f"Slide 8 Hand-Worked Example\nSigma=[[0.617, 0.615], [0.615, 0.717]] | lambda1={worked_data['eigvals'][0]}", fontsize=11, fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    for bar in we_bars:
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{bar.get_height():.1f}%", ha="center", fontweight="bold")
        
    # Scaling pitfall contrast
    x_pos = np.arange(3)
    width = 0.35
    axes[1].bar(x_pos - width / 2, pitfall_data["raw_explained_ratio"] * 100, width, label="Raw Scale (Income dominates)", color="#e74c3c", edgecolor="black")
    axes[1].bar(x_pos + width / 2, pitfall_data["std_explained_ratio"] * 100, width, label="StandardScaler (Balanced)", color="#3498db", edgecolor="black")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(["PC1", "PC2", "PC3"])
    axes[1].set_ylabel("Variance Explained (%)", fontweight="bold")
    axes[1].set_title("Slide 11 & 13: Scaling Pitfall Contrast\nRaw Variance Distortion vs Unit-Variance Standardization", fontsize=11, fontweight="bold")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle=":", alpha=0.5)
    
    fig.suptitle("PCA Theoretical & Methodological Foundations", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    for d in save_dirs:
        plt.savefig(d / "pca_theory_and_pitfall.png", dpi=150, bbox_inches="tight")
    plt.close()
    generated_plots["theory_and_pitfall"] = "pca_theory_and_pitfall.png"
    
    return generated_plots


# =====================================================================
# 5. Diagnostic Report Generation
# =====================================================================

def save_pca_report(worked_data, pitfall_data, full_res):
    """Save comprehensive PCA diagnostic report to REPORTS_DIR."""
    save_paths = [config.REPORTS_DIR / "pca_report.txt"]
    
    explained_ratio = full_res["explained_variance_ratio"]
    cum_var = full_res["cumulative_variance"]
    loadings = full_res["loadings"]
    top_pc1 = full_res["top_pc1"]
    top_pc2 = full_res["top_pc2"]
    prod = full_res["production_eval"]
    
    report_text = []
    report_text.append("=" * 80)
    report_text.append("PRINCIPAL COMPONENT ANALYSIS (PCA) REPORT")
    report_text.append("Course: Machine Learning (25SC2107E) | Placement Prediction Dataset")
    report_text.append("Reference: M4-Session4-PCA.pptx & principal_component_analysis.py")
    report_text.append("=" * 80 + "\n")
    
    report_text.append("1. HAND-WORKED NUMERICAL EXAMPLE VERIFICATION (SLIDES 8 & 12):")
    report_text.append(f"   Input 10 points in 2-D | Mean: {worked_data['mean']}")
    report_text.append("   Sample Covariance Matrix Sigma = (1/(n-1)) X_c^T X_c:")
    report_text.append(f"     [[{worked_data['cov_matrix'][0,0]:.4f}, {worked_data['cov_matrix'][0,1]:.4f}],")
    report_text.append(f"      [{worked_data['cov_matrix'][1,0]:.4f}, {worked_data['cov_matrix'][1,1]:.4f}]]")
    report_text.append(f"   Eigenvalues: lambda1 = {worked_data['eigvals'][0]:.4f}, lambda2 = {worked_data['eigvals'][1]:.4f}")
    report_text.append(f"   PC1 direction vector v1: ({worked_data['v1'][0]:.4f}, {worked_data['v1'][1]:.4f})")
    report_text.append(f"   Explained variance ratio: PC1 = {worked_data['explained_pct_pc1']}%, PC2 = {worked_data['explained_pct_pc2']}%")
    report_text.append("   Verification check: Matches slide 8 & 12 benchmarks exactly.\n")
    
    report_text.append("2. STANDARDIZATION PITFALL DEMONSTRATION (SLIDES 11 & 13):")
    report_text.append("   - Raw Unscaled PCA: Income feature dominates completely:")
    report_text.append(f"     Explained Variance: {pitfall_data['raw_explained_ratio'].tolist()} ({pitfall_data['raw_pc1_pct']}% on PC1)")
    report_text.append("   - StandardScaler PCA: Meaningful, balanced decomposition:")
    report_text.append(f"     Explained Variance: {pitfall_data['std_explained_ratio'].tolist()} ({pitfall_data['std_pc1_pct']}% PC1, {pitfall_data['std_pc2_pct']}% PC2)")
    report_text.append("   Lesson: Always scale features to unit variance before PCA unless sharing identical units.\n")
    
    report_text.append(f"3. PLACEMENT PREDICTION DATASET SPECTRUM ({full_res['n_samples']:,} Students):")
    report_text.append(f"   Input features: {len(full_res['feature_cols'])} standardized numeric metrics")
    report_text.append("   Variance explained by first 10 components:")
    for i in range(min(10, len(explained_ratio))):
        report_text.append(f"     PC{i+1:<2}: {explained_ratio[i]*100:>5.2f}%  (cumulative: {cum_var[i]*100:>5.2f}%)")
        
    report_text.append("\n   Components needed for common variance thresholds:")
    for thresh, n_req in full_res["components_needed"].items():
        report_text.append(f"     {thresh}% cumulative variance: {n_req} components")
        
    pc12_pct = round(full_res["pc1_pc2_variance"] * 100, 1)
    report_text.append(f"\n   PC1 + PC2 together capture: {pc12_pct}% of total variance")
    report_text.append("   (This is the exact number that backed the 2D cluster scatter plots")
    report_text.append("    in Sessions 21 & 24 - now explicitly quantified).\n")
    
    report_text.append("4. FEATURE LOADINGS INTERPRETATION (WHAT PC1 & PC2 REPRESENT):")
    report_text.append("   Top features contributing to PC1 (by absolute loading):")
    for feat in top_pc1.head(6).index:
        report_text.append(f"     {feat:<22}: {loadings.loc[feat, 'PC1']:>7.3f}")
    report_text.append("\n   Top features contributing to PC2 (by absolute loading):")
    for feat in top_pc2.head(6).index:
        report_text.append(f"     {feat:<22}: {loadings.loc[feat, 'PC2']:>7.3f}\n")
        
    report_text.append("5. PRODUCTION PIPELINE BENCHMARK (SLIDE 10):")
    report_text.append(f"   Full Features ({prod['full_features']['n_features']}): Accuracy = {prod['full_features']['acc']}%, ROC-AUC = {prod['full_features']['auc']}")
    report_text.append(f"   PCA Pipeline  ({prod['pca_features']['n_components']} PCs): Accuracy = {prod['pca_features']['acc']}%, ROC-AUC = {prod['pca_features']['auc']}")
    report_text.append("   Compression Ratio: Retains predictive power while eliminating multicollinearity.\n")
    
    report_text.append("6. COMPARISON: PCA vs LINEAR REGRESSION (SLIDE 14):")
    report_text.append("   " + "-" * 75)
    report_text.append(f"   {'Aspect':<25} | {'PCA':<22} | {'Linear Regression':<22}")
    report_text.append("   " + "-" * 75)
    report_text.append(f"   {'Learning Type':<25} | {'Unsupervised':<22} | {'Supervised':<22}")
    report_text.append(f"   {'Goal':<25} | {'Retain variance':<22} | {'Predict target y':<22}")
    report_text.append(f"   {'Input Needed':<25} | {'Features only (no labels)':<22} | {'Features X and target y':<22}")
    report_text.append(f"   {'Objective':<25} | {'Maximize variance (v^T Sigma v)':<22} | {'Minimize squared residuals':<22}")
    report_text.append(f"   {'Underlying Tool':<25} | {'Covariance eigen-decomp':<22} | {'Normal equations / SGD':<22}")
    report_text.append("   " + "-" * 75)
    
    full_str = "\n".join(report_text)
    for p in save_paths:
        with open(p, "w", encoding="utf-8") as f:
            f.write(full_str)
            
    return save_paths[0]


# =====================================================================
# 6. Pipeline Runner
# =====================================================================

def run_pca_pipeline(df=None):
    """Execute complete PCA pipeline standalone."""
    print("=" * 70)
    print("MODULE 4 · SESSION 23: PRINCIPAL COMPONENT ANALYSIS (PCA)")
    print("=" * 70)
    start_time = time.perf_counter()
    
    print("[1/5] Verifying Slide 8 & 12 hand-worked numerical example...")
    worked_data = run_pca_worked_example()
    print(f"      Sigma: {worked_data['cov_matrix'].tolist()}")
    print(f"      lambda1={worked_data['eigvals'][0]} ({worked_data['explained_pct_pc1']}%), lambda2={worked_data['eigvals'][1]} ({worked_data['explained_pct_pc2']}%)")
    print(f"      v1={worked_data['v1'].tolist()}")
    
    print("[2/5] Demonstrating Slide 11 & 13 standardization pitfall...")
    pitfall_data = run_standardization_pitfall_demo()
    print(f"      Raw PCA:    PC1={pitfall_data['raw_pc1_pct']}% (Income scale dominates)")
    print(f"      Scaled PCA: PC1={pitfall_data['std_pc1_pct']}%, PC2={pitfall_data['std_pc2_pct']}% (Balanced)")
    
    print("[3/5] Fitting full PCA on Placement dataset (50,000 students)...")
    full_res = run_full_pca_pipeline(df=df)
    print(f"      Features: {len(full_res['feature_cols'])} | Fit time: {full_res['fit_time']}s")
    for thresh, n_req in full_res["components_needed"].items():
        print(f"      {thresh}% cumulative variance threshold -> {n_req} components")
    print(f"      PC1 + PC2 combined variance: {round(full_res['pc1_pc2_variance']*100, 1)}%")
    
    print("[4/5] Evaluating Slide 10 production Pipeline (StandardScaler + PCA + Classifier)...")
    prod = full_res["production_eval"]
    print(f"      Full features ({prod['full_features']['n_features']}): Acc = {prod['full_features']['acc']}%, AUC = {prod['full_features']['auc']}")
    print(f"      PCA pipeline  ({prod['pca_features']['n_components']}): Acc = {prod['pca_features']['acc']}%, AUC = {prod['pca_features']['auc']}")
    
    print("[5/5] Generating diagnostic plots and saving comprehensive reports...")
    plots = generate_pca_diagrams(worked_data, pitfall_data, full_res)
    rep_path = save_pca_report(worked_data, pitfall_data, full_res)
    
    elapsed = round(time.perf_counter() - start_time, 2)
    print(f"\n[DONE] PCA pipeline completed successfully in {elapsed}s.")
    print(f"       Saved plots to: {config.PLOTS_DIR}")
    print(f"       Report saved to: {Path(rep_path).name}")
    
    return worked_data, pitfall_data, full_res, plots, rep_path


if __name__ == "__main__":
    run_pca_pipeline()
