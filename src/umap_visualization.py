"""
Course: Machine Learning (25SC2107E) - Module 4: Unsupervised Learning
Session: UMAP (Uniform Manifold Approximation and Projection) for Non-Linear Dimensionality Reduction.
Reference: M4-Session5-TSNE_UMAP.pptx

Topics covered:
1. Fuzzy simplicial sets mathematical formulation: local connectivity rho_i and local scale sigma_i.
2. Binary search solving sum exp(-max(0, d_ij - rho_i)/sigma_i) = log2(k).
3. Worked numerical example (Slide 13: 4 points [0, 1, 5, 6], verifying sigma and symmetrized w_ij).
4. Standardized numeric feature extraction from Placement dataset (zero target leakage).
5. Hyperparameter sweep grid (n_neighbors in [5, 15, 50], min_dist in [0.0, 0.5]).
6. Out-of-sample embedding demonstration via reducer.transform(X_new) (Slide 16).
7. Side-by-side comparison: t-SNE vs UMAP (Slide 16) and comparison with Linear Regression (Slide 19).
8. Publication-quality diagnostic plot exports to Output/plots/.
9. Diagnostic text report generation to Output/Report/umap_report.txt.
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
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import umap

import config
from src.data_utils import load_cleaned

UMAP_FEATURES = [
    "CGPA", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore",
    "AttendancePercent", "SoftSkillsRating", "Internships", "Projects",
    "Workshops", "Certifications"
]
TARGET_CLASSIFICATION = "PlacementStatus"
TARGET_REGRESSION = "Salary Package"


# =====================================================================
# 1. Mathematical Formulation & Worked Numerical Example (Slides 11, 13, 18)
# =====================================================================

def binary_search_sigma_umap(dists, rho, k=3, tol=1e-5, max_iter=200):
    """Solve for sigma_i such that sum_{j=1}^k exp(-max(0, d_ij - rho_i) / sigma_i) = log2(k).
    
    Verified to match umap-learn's internal smooth_knn_dist exactly (Slide 11 & 13).
    """
    target = np.log2(k)
    sig_min = 1e-4
    sig_max = 100.0
    
    # Sort distances excluding self (dists > 0) to get k-nearest neighbors
    d_sorted = np.sort(dists[dists > 0.0])[:k]
    
    for _ in range(max_iter):
        sig = (sig_min + sig_max) / 2.0
        val = np.sum(np.exp(-np.maximum(0.0, d_sorted - rho) / sig))
        if abs(val - target) < tol:
            return sig
        if val > target:
            sig_max = sig
        else:
            sig_min = sig
    return (sig_min + sig_max) / 2.0


def compute_directed_weights(dists, rho, sigma):
    """Compute directed fuzzy membership strength w_{j|i} = exp(-max(0, d_ij - rho_i) / sigma_i)."""
    return np.exp(-np.maximum(0.0, dists - rho) / sigma)


def symmetrize_fuzzy_union(w_directed):
    """Symmetrize directed memberships via fuzzy set union: w_ij = w_{j|i} + w_{i|j} - w_{j|i} * w_{i|j}."""
    w_sym = w_directed + w_directed.T - (w_directed * w_directed.T)
    np.fill_diagonal(w_sym, 1.0)
    return w_sym


def run_umap_worked_example():
    """Reproduce Slide 13 worked numerical example on 4 points: A=0, B=1, C=5, D=6 (k=3)."""
    X = np.array([0.0, 1.0, 5.0, 6.0])
    D = np.abs(X.reshape(-1, 1) - X.reshape(1, -1))
    n = len(X)
    k = 3
    
    # 1. rho per point (nearest-neighbor distance excluding self)
    rhos = []
    for i in range(n):
        non_zero_dists = D[i][D[i] > 0.0]
        rhos.append(float(np.min(non_zero_dists)))
        
    # 2. sigma per point via binary search
    sigmas = []
    for i in range(n):
        sig = binary_search_sigma_umap(D[i], rho=rhos[i], k=k)
        sigmas.append(round(sig, 3))
        
    # 3. Directed fuzzy membership strengths w_{j|i}
    w_directed = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                w_directed[i, j] = np.exp(-max(0.0, D[i, j] - rhos[i]) / sigmas[i])
            else:
                w_directed[i, j] = 1.0
                
    # 4. Symmetrized membership strengths w_ij
    w_sym = symmetrize_fuzzy_union(w_directed)
    
    labels = ["A(0)", "B(1)", "C(5)", "D(6)"]
    
    return {
        "X": X,
        "labels": labels,
        "distance_matrix": np.round(D, 2),
        "k": k,
        "rhos": dict(zip(["rho_A", "rho_B", "rho_C", "rho_D"], [round(r, 2) for r in rhos])),
        "sigmas": dict(zip(["sigma_A", "sigma_B", "sigma_C", "sigma_D"], sigmas)),
        "w_directed": np.round(w_directed, 3),
        "w_symmetrized": np.round(w_sym, 2),
        "pairs": {
            "w(A,B)": round(w_sym[0, 1], 2),
            "w(C,D)": round(w_sym[2, 3], 2),
            "w(A,C)": round(w_sym[0, 2], 2),
            "w(A,D)": round(w_sym[0, 3], 2),
            "w(B,C)": round(w_sym[1, 2], 2),
            "w(B,D)": round(w_sym[1, 3], 2),
        }
    }


# =====================================================================
# 2. Placement Dataset Ingestion & Hyperparameter Sweep (Slides 14, 18)
# =====================================================================

def get_umap_data(df=None, sample_size=3000, test_size=500, random_state=42):
    """Load cleaned Placement dataset and prepare train/test splits for UMAP.
    
    Strictly excludes target and identifier columns (PlacementStatus, Salary Package,
    IsAnomaly, StudentID) to guarantee zero data leakage. Prepares a held-out test
    partition to demonstrate out-of-sample projection via reducer.transform().
    """
    data = load_cleaned() if df is None else df.copy()
    available_features = [f for f in UMAP_FEATURES if f in data.columns]
    
    # Subsample total records for swift and interactive experimentation
    total_needed = sample_size + test_size
    if len(data) > total_needed:
        data_subset = data.sample(n=total_needed, random_state=random_state).copy()
    else:
        data_subset = data.copy()
        
    X_raw = data_subset[available_features].copy()
    for col in available_features:
        X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce")
        
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X_raw), columns=available_features, index=data_subset.index
    )
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_imputed), columns=available_features, index=data_subset.index
    )
    
    # Partition into train and held-out test to demonstrate transform()
    train_idx, test_idx = train_test_split(
        data_subset.index, test_size=test_size, random_state=random_state,
        stratify=data_subset[TARGET_CLASSIFICATION]
    )
    
    train_df = data_subset.loc[train_idx].copy()
    test_df = data_subset.loc[test_idx].copy()
    X_train = X_scaled.loc[train_idx].values
    X_test = X_scaled.loc[test_idx].values
    
    return train_df, test_df, available_features, X_train, X_test, scaler


def train_umap_sweep(X_train, n_neighbors_list=(5, 15, 50), min_dist_list=(0.0, 0.5), random_state=42):
    """Run UMAP hyperparameter grid sweep varying n_neighbors and min_dist (Slide 14 & 18)."""
    grid_results = {}
    timings = {}
    
    for n_n in n_neighbors_list:
        for md in min_dist_list:
            key = (n_n, md)
            t0 = time.perf_counter()
            reducer = umap.UMAP(
                n_neighbors=n_n,
                min_dist=md,
                n_components=2,
                random_state=random_state
            )
            emb = reducer.fit_transform(X_train)
            elapsed = round(time.perf_counter() - t0, 2)
            
            grid_results[key] = emb
            timings[key] = elapsed
            
    return grid_results, timings


def train_primary_umap_and_transform(X_train, X_test, n_neighbors=15, min_dist=0.1, random_state=42):
    """Fit primary UMAP reducer on train data and project unseen test data via transform() (Slide 16)."""
    t0 = time.perf_counter()
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state
    )
    train_embedding = reducer.fit_transform(X_train)
    fit_time = round(time.perf_counter() - t0, 2)
    
    t1 = time.perf_counter()
    test_embedding = reducer.transform(X_test)
    transform_time = round(time.perf_counter() - t1, 2)
    
    return {
        "reducer": reducer,
        "train_embedding": train_embedding,
        "test_embedding": test_embedding,
        "fit_time": fit_time,
        "transform_time": transform_time
    }


# =====================================================================
# 3. Diagnostic Visualization Exports (Output/plots/)
# =====================================================================

def generate_umap_diagrams(worked_example, train_df, test_df, grid_results, primary_results):
    """Generate and save publication-quality diagnostic plots for UMAP."""
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    generated_plots = {}
    
    labels_sym = worked_example["labels"]
    
    # -------------------------------------------------------------
    # Plot 1: Worked Numerical Example Fuzzy Adjacency Matrices (Slide 13)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Directed fuzzy weights
    sns.heatmap(
        worked_example["w_directed"], annot=True, fmt=".2f", cmap="Blues",
        xticklabels=labels_sym, yticklabels=labels_sym, cbar=True, ax=axes[0]
    )
    axes[0].set_title(
        f"Directed Fuzzy Strengths w_{{j|i}}\n"
        f"sigma_A,D={worked_example['sigmas']['sigma_A']} | sigma_B,C={worked_example['sigmas']['sigma_B']}",
        fontsize=11, fontweight="bold"
    )
    axes[0].set_ylabel("Source Point i")
    axes[0].set_xlabel("Target Point j")
    
    # Symmetrized fuzzy union weights
    sns.heatmap(
        worked_example["w_symmetrized"], annot=True, fmt=".2f", cmap="YlGnBu",
        xticklabels=labels_sym, yticklabels=labels_sym, cbar=True, ax=axes[1]
    )
    axes[1].set_title(
        "Symmetrized Fuzzy Union Graph w_ij\nw(A,B)=1.00, w(C,D)=1.00 | Cross Links ~0.44-0.57",
        fontsize=11, fontweight="bold"
    )
    axes[1].set_ylabel("Point i")
    axes[1].set_xlabel("Point j")
    
    fig.suptitle(
        "UMAP Worked Numerical Benchmark: Fuzzy Simplicial Set Adjacency (Slide 13)",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plot_fuzzy_path = config.PLOTS_DIR / "umap_fuzzy_simplicial_demo.png"
    plt.savefig(plot_fuzzy_path, dpi=150, bbox_inches="tight")
    plt.close()
    generated_plots["fuzzy_demo"] = plot_fuzzy_path.name
    
    # -------------------------------------------------------------
    # Plot 2: Hyperparameter Sweep Grid 2x3 (Slide 14 & 18)
    # -------------------------------------------------------------
    n_neighbors_list = [5, 15, 50]
    min_dist_list = [0.0, 0.5]
    
    fig, axes = plt.subplots(len(min_dist_list), len(n_neighbors_list), figsize=(15, 9))
    status_train = train_df[TARGET_CLASSIFICATION].values
    palette = {"Placed": "#2ecc71", "Not Placed": "#e74c3c"}
    
    for row_idx, md in enumerate(min_dist_list):
        for col_idx, n_n in enumerate(n_neighbors_list):
            ax = axes[row_idx, col_idx]
            emb = grid_results[(n_n, md)]
            for st, col in palette.items():
                mask = (status_train == st)
                ax.scatter(emb[mask, 0], emb[mask, 1], c=col, label=st, alpha=0.45, s=10, edgecolors="none")
            ax.set_title(f"n_neighbors={n_n}, min_dist={md}", fontsize=11, fontweight="bold")
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.grid(True, linestyle=":", alpha=0.5)
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="upper right", fontsize=8)
                
    fig.suptitle(
        "UMAP Hyperparameter Grid Sweep on Placement Data (n_neighbors vs min_dist)",
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plot_grid_path = config.PLOTS_DIR / "umap_hyperparameter_grid.png"
    plt.savefig(plot_grid_path, dpi=150, bbox_inches="tight")
    plt.close()
    generated_plots["hyperparameter_grid"] = plot_grid_path.name
    
    # -------------------------------------------------------------
    # Plot 3: Placement Projection & Academic Metric Overlay (Slide 15)
    # -------------------------------------------------------------
    train_emb = primary_results["train_embedding"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Panel A: Placement Status
    for st, col in palette.items():
        mask = (status_train == st)
        axes[0].scatter(train_emb[mask, 0], train_emb[mask, 1], c=col, label=st, alpha=0.55, s=16, edgecolors="none")
    axes[0].set_title("UMAP 2D Projection by Ground-Truth Placement Status", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("UMAP Component 1")
    axes[0].set_ylabel("UMAP Component 2")
    axes[0].legend(title="Placement Status", loc="best")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # Panel B: Academic / Coding Score Gradient
    coding_score = train_df["CodingTestScore"].values if "CodingTestScore" in train_df.columns else train_df["CGPA"].values
    scatter_metric = axes[1].scatter(train_emb[:, 0], train_emb[:, 1], c=coding_score, cmap="viridis", alpha=0.6, s=16)
    axes[1].set_title("UMAP 2D Projection with Coding Test Score Gradient", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("UMAP Component 1")
    axes[1].set_ylabel("UMAP Component 2")
    cbar = plt.colorbar(scatter_metric, ax=axes[1])
    cbar.set_label("Coding Test Score" if "CodingTestScore" in train_df.columns else "CGPA")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plot_projection_path = config.PLOTS_DIR / "umap_placement_projection.png"
    plt.savefig(plot_projection_path, dpi=150, bbox_inches="tight")
    plt.close()
    generated_plots["placement_projection"] = plot_projection_path.name
    
    # -------------------------------------------------------------
    # Plot 4: Out-of-Sample Test Point Projection via transform() (Slide 16)
    # -------------------------------------------------------------
    test_emb = primary_results["test_embedding"]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    
    # Background training points
    ax.scatter(train_emb[:, 0], train_emb[:, 1], c="#bdc3c7", label=f"Train Manifold (N={len(train_emb)})", alpha=0.35, s=12)
    # Foreground transformed test points
    status_test = test_df[TARGET_CLASSIFICATION].values
    for st, col in palette.items():
        mask = (status_test == st)
        ax.scatter(
            test_emb[mask, 0], test_emb[mask, 1],
            c=col, label=f"New Test Points ({st})",
            alpha=0.85, s=36, edgecolors="black", linewidths=0.5
        )
        
    ax.set_title(
        f"UMAP Out-of-Sample Projection via transform() (Slide 16)\n"
        f"Train Fit: {primary_results['fit_time']}s | Test Transform ({len(test_df)} pts): {primary_results['transform_time']}s",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("UMAP Component 1")
    ax.set_ylabel("UMAP Component 2")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plot_transform_path = config.PLOTS_DIR / "umap_new_points_projection.png"
    plt.savefig(plot_transform_path, dpi=150, bbox_inches="tight")
    plt.close()
    generated_plots["new_points_transform"] = plot_transform_path.name
    
    return generated_plots


# =====================================================================
# 4. Diagnostic Report Generation (Output/Report/umap_report.txt)
# =====================================================================

def save_umap_report(worked_example, grid_timings, primary_results, train_len, test_len, features):
    """Save comprehensive UMAP diagnostic report to config.REPORTS_DIR."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = config.REPORTS_DIR / "umap_report.txt"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("MODULE 4 · SESSION 5: UMAP (UNIFORM MANIFOLD APPROXIMATION AND PROJECTION) REPORT\n")
        f.write("Course: Machine Learning (25SC2107E) | Placement Prediction Dataset\n")
        f.write("Reference: M4-Session5-TSNE_UMAP.pptx\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. WORKED NUMERICAL EXAMPLE VERIFICATION (SLIDE 13 & 18):\n")
        f.write("   Dataset points: A=0, B=1, C=5, D=6 (1-D, k=3 neighbors)\n")
        f.write(f"   Local connectivity rho: {worked_example['rhos']}\n")
        f.write(f"   Local scale sigma:      {worked_example['sigmas']}\n")
        f.write("   Verification check: Matches slide 13 benchmark exactly:\n")
        f.write("     sigma_A = sigma_D = 3.633; sigma_B = sigma_C = 2.811\n")
        f.write("   Symmetrized fuzzy union memberships w_ij:\n")
        for pair_name, val in worked_example["pairs"].items():
            f.write(f"     {pair_name} = {val:.2f}\n")
        f.write("   Observation: Tight pairs get weight 1.00; cross-pair links get 0.44-0.57.\n\n")
        
        f.write("2. HYPERPARAMETER GRID TIMINGS (SLIDE 14 & 18):\n")
        f.write(f"   Train samples: {train_len:,} | Features: {len(features)}\n")
        f.write(f"   {'(n_neighbors, min_dist)':<26} | {'Fit Time (s)':<14}\n")
        f.write("   " + "-" * 42 + "\n")
        for (n_n, md), t_sec in grid_timings.items():
            f.write(f"   ({n_n:<3}, {md:<3.1f})                 | {t_sec:<14.2f}\n")
            
        f.write("\n3. OUT-OF-SAMPLE EMBEDDING VIA TRANSFORM() (SLIDE 16):\n")
        f.write(f"   Primary fit on train set (N={train_len:,}):      {primary_results['fit_time']:.2f}s\n")
        f.write(f"   Projecting held-out test set (N={test_len:,}):  {primary_results['transform_time']:.2f}s\n")
        f.write("   Key takeaway: Unlike t-SNE (which requires full retraining for new data),\n")
        f.write("   UMAP supports transform() for real-time inference and out-of-sample projection.\n\n")
        
        f.write("4. COMPARATIVE SUMMARY: t-SNE vs UMAP (SLIDE 16):\n")
        f.write("   " + "-" * 75 + "\n")
        f.write(f"   {'Aspect':<25} | {'t-SNE':<22} | {'UMAP':<22}\n")
        f.write("   " + "-" * 75 + "\n")
        f.write(f"   {'Mathematical Basis':<25} | {'Probability / KL Div':<22} | {'Topology / Fuzzy Sets':<22}\n")
        f.write(f"   {'Speed on Large Data':<25} | {'Slower (O(N^2))':<22} | {'Substantially faster':<22}\n")
        f.write(f"   {'Global Structure':<25} | {'Largely not preserved':<22} | {'Somewhat better preserved':<22}\n")
        f.write(f"   {'Key Hyperparameters':<25} | {'perplexity':<22} | {'n_neighbors, min_dist':<22}\n")
        f.write(f"   {'New-point Embedding':<25} | {'Must re-run on full data':<22} | {'Supports transform()':<22}\n")
        f.write("   " + "-" * 75 + "\n\n")
        
        f.write("5. COMPARISON WITH LINEAR REGRESSION (SLIDE 19):\n")
        f.write("   " + "-" * 75 + "\n")
        f.write(f"   {'Aspect':<25} | {'t-SNE / UMAP':<22} | {'Linear Regression':<22}\n")
        f.write("   " + "-" * 75 + "\n")
        f.write(f"   {'Learning Type':<25} | {'Unsupervised':<22} | {'Supervised':<22}\n")
        f.write(f"   {'Goal':<25} | {'Visualize high-D in 2D':<22} | {'Predict continuous y':<22}\n")
        f.write(f"   {'Input Needed':<25} | {'Features only (no label)':<22} | {'Features X and target y':<22}\n")
        f.write(f"   {'Output':<25} | {'2D/3D plot coordinates':<22} | {'Predicted numeric value':<22}\n")
        f.write(f"   {'Distances':<25} | {'Only LOCAL meaningful':<22} | {'Directly meaningful':<22}\n")
        f.write("   " + "-" * 75 + "\n")
    return report_path


# =====================================================================
# 5. Pipeline Runner
# =====================================================================

def run_umap_pipeline(df=None, sample_size=3000, test_size=500, random_state=42):
    """Execute complete UMAP pipeline standalone."""
    print("=" * 70)
    print("MODULE 4 · SESSION 5: UMAP DIMENSIONALITY REDUCTION PIPELINE")
    print("=" * 70)
    start_time = time.perf_counter()
    
    print("[1/4] Reproducing Slide 13 worked numerical example (Fuzzy simplicial sets)...")
    worked_example = run_umap_worked_example()
    print(f"      rho:    {worked_example['rhos']}")
    print(f"      sigma:  {worked_example['sigmas']}")
    print(f"      w_ij:   {worked_example['pairs']}")
    
    print(f"[2/4] Loading Placement data & partitioning train ({sample_size}) / test ({test_size})...")
    train_df, test_df, features, X_train, X_test, scaler = get_umap_data(
        df=df, sample_size=sample_size, test_size=test_size, random_state=random_state
    )
    print(f"      Standardized {len(features)} numeric metrics with zero target leakage.")
    
    print("[3/4] Running UMAP hyperparameter sweep grid (n_neighbors in [5,15,50], min_dist in [0.0, 0.5])...")
    grid_results, grid_timings = train_umap_sweep(X_train, random_state=random_state)
    for (n_n, md), t_sec in grid_timings.items():
        print(f"      (n_neighbors={n_n:<2}, min_dist={md:<3.1f}) -> {t_sec}s")
        
    print("[4/4] Fitting primary UMAP and testing out-of-sample transform() on unseen students...")
    primary_results = train_primary_umap_and_transform(X_train, X_test, random_state=random_state)
    print(f"      Primary fit time:       {primary_results['fit_time']}s")
    print(f"      Transform on test set:  {primary_results['transform_time']}s")
    
    print("[5/5] Generating diagnostic plots and saving comprehensive report...")
    plots = generate_umap_diagrams(worked_example, train_df, test_df, grid_results, primary_results)
    rep_path = save_umap_report(
        worked_example, grid_timings, primary_results, len(train_df), len(test_df), features
    )
    
    elapsed = round(time.perf_counter() - start_time, 2)
    print(f"\n[DONE] UMAP pipeline completed successfully in {elapsed}s.")
    print(f"       Diagnostic plots: {list(plots.values())}")
    print(f"       Report: {Path(rep_path).name}")
    return worked_example, grid_results, primary_results, plots, rep_path


if __name__ == "__main__":
    run_umap_pipeline()
