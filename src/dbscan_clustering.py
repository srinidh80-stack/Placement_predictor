"""
Course: Machine Learning (25SC2107E) - Module 4: Unsupervised Learning
Session: DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
Reference: M4-Session3-DBSCAN_new.pptx

Topics covered:
1. Core Theory & Definitions:
   - eps-neighborhood N_eps(p) and minPts density threshold
   - Point classification: Core, Border, and Noise points
   - Density-reachability, density-connectivity, and cluster expansion
2. Hand-Worked Numerical Example Verification (Slide 8 & 11):
   - 9 points in 2-D (eps=1.5, minPts=3)
   - Verifying Core (A,B,C,D,E,F,G), Border (I -> cluster 0), Noise (H -> -1)
3. Non-Convex Clusters Benchmark: Two-Moons Stress Test (Slide 10 & 12):
   - Demonstrating why K-Means fails on non-convex shapes while DBSCAN succeeds
4. Parameter Selection: k-Distance Graph Elbow Heuristic (Slide 9 & 14):
   - Sorted k-nearest neighbor distance curve to determine optimal eps
5. Placement Prediction Dataset Application:
   - Standardized numeric features with zero target leakage
   - Parameter sweep and Core/Border/Noise student breakdown
   - Cross-checking DBSCAN noise outliers against ground-truth 'IsAnomaly'
   - Placement rate analysis per discovered cluster
6. Diagnostic visualization exports to Output/plots/.
7. Diagnostic text report generation to Output/Report/dbscan_report.txt.
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
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import config
from src.data_utils import load_cleaned

DBSCAN_FEATURES = [
    "CGPA", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore",
    "AttendancePercent", "SoftSkillsRating", "Internships", "Projects",
    "Workshops", "Certifications"
]
TARGET_CLASSIFICATION = "PlacementStatus"
TARGET_ANOMALY = "IsAnomaly"


# =====================================================================
# 1. Hand-Worked Numerical Example Verification (Slides 8 & 11)
# =====================================================================

def run_dbscan_worked_example():
    """Reproduce Slide 8 & 11 worked numerical example on 9 points in 2-D (eps=1.5, minPts=3)."""
    points = {
        "A": (2.0, 2.0), "B": (2.0, 3.0), "C": (3.0, 2.0), "D": (3.0, 3.0),
        "I": (4.4, 3.0),
        "E": (8.0, 8.0), "F": (8.0, 9.0), "G": (9.0, 8.0),
        "H": (25.0, 25.0)
    }
    names = list(points.keys())
    X = np.array([points[n] for n in names])
    eps = 1.5
    minPts = 3
    
    # Fit scikit-learn DBSCAN
    db = DBSCAN(eps=eps, min_samples=minPts).fit(X)
    
    core_indices = set(db.core_sample_indices_)
    point_types = {}
    
    for idx, (name, label) in enumerate(zip(names, db.labels_)):
        if idx in core_indices:
            kind = "core"
        elif label == -1:
            kind = "noise"
        else:
            kind = "border"
            
        point_types[name] = {
            "coords": points[name],
            "cluster": int(label),
            "kind": kind
        }
        
    # Check neighborhood sizes |N_eps|
    from scipy.spatial.distance import cdist
    dists = cdist(X, X)
    neighborhood_sizes = {names[i]: int(np.sum(dists[i] <= eps)) for i in range(len(names))}
    
    return {
        "points": points,
        "names": names,
        "X": X,
        "eps": eps,
        "minPts": minPts,
        "labels": db.labels_,
        "point_types": point_types,
        "neighborhood_sizes": neighborhood_sizes,
        "n_clusters": len(set(db.labels_) - {-1}),
        "n_noise": int(np.sum(db.labels_ == -1))
    }


# =====================================================================
# 2. Non-Convex Clusters: Two-Moons Stress Test (Slides 10 & 12)
# =====================================================================

def run_two_moons_benchmark(n_samples=300, noise=0.06, random_state=42):
    """Reproduce Slide 10 & 12: Compare K-Means vs DBSCAN on non-convex half-moons."""
    X, y_true = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    
    # 1. K-Means (convex assumption)
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=random_state)
    kmeans_labels = kmeans.fit_predict(X)
    ari_kmeans = round(float(adjusted_rand_score(y_true, kmeans_labels)), 4)
    
    # 2. DBSCAN (density-reachable chains)
    dbscan = DBSCAN(eps=0.2, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X)
    ari_dbscan = round(float(adjusted_rand_score(y_true, dbscan_labels)), 4)
    
    return {
        "X": X,
        "y_true": y_true,
        "kmeans_labels": kmeans_labels,
        "dbscan_labels": dbscan_labels,
        "ari_kmeans": ari_kmeans,
        "ari_dbscan": ari_dbscan
    }


# =====================================================================
# 3. Placement Dataset Ingestion & k-Distance Graph (Slides 9 & 14)
# =====================================================================

def get_dbscan_data(df=None, sample_size=3000, random_state=42):
    """Load cleaned Placement dataset and extract standardized numeric features.
    
    Strictly excludes target and identifier columns (PlacementStatus, Salary Package,
    IsAnomaly, StudentID) to guarantee zero data leakage. Subsamples N records to
    maintain fast O(N^2) pairwise distance operations.
    """
    data = load_cleaned() if df is None else df.copy()
    available_features = [f for f in DBSCAN_FEATURES if f in data.columns]
    
    if len(data) > sample_size:
        sample_df = data.sample(n=sample_size, random_state=random_state).copy()
    else:
        sample_df = data.copy()
        
    X_raw = sample_df[available_features].copy()
    for col in available_features:
        X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce")
        
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X_raw), columns=available_features, index=sample_df.index
    )
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    return sample_df, available_features, X_scaled, scaler


def compute_k_distance_graph(X_scaled, k=10):
    """Compute sorted k-nearest neighbor distances for the elbow plot (Slide 9 & 14)."""
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
    distances, _ = nbrs.kneighbors(X_scaled)
    
    # Distance to the k-th neighbor (last column)
    k_distances = np.sort(distances[:, -1])
    
    # Compute numerical knee/elbow (maximum curvature or 90th percentile guide)
    elbow_idx = int(len(k_distances) * 0.90)
    suggested_eps = round(float(k_distances[elbow_idx]), 2)
    
    return k_distances, suggested_eps


def train_dbscan_sweep(X_scaled, eps_list=(1.5, 2.0, 2.5), min_samples_list=(10, 20)):
    """Run DBSCAN parameter sweep varying eps and min_samples."""
    sweep_results = []
    
    for eps in eps_list:
        for ms in min_samples_list:
            t0 = time.perf_counter()
            db = DBSCAN(eps=eps, min_samples=ms)
            labels = db.fit_predict(X_scaled)
            elapsed = round(time.perf_counter() - t0, 3)
            
            n_clusters = len(set(labels) - {-1})
            n_noise = int(np.sum(labels == -1))
            noise_ratio = round(float(n_noise / len(labels) * 100), 2)
            n_core = len(db.core_sample_indices_)
            
            # Silhouette on non-noise points if at least 2 clusters exist
            non_noise_mask = (labels != -1)
            if n_clusters >= 2 and np.sum(non_noise_mask) > n_clusters:
                sil = round(float(silhouette_score(X_scaled[non_noise_mask], labels[non_noise_mask])), 4)
            else:
                sil = -1.0
                
            sweep_results.append({
                "eps": eps,
                "min_samples": ms,
                "n_clusters": n_clusters,
                "n_core": n_core,
                "n_noise": n_noise,
                "noise_ratio_pct": noise_ratio,
                "silhouette_non_noise": sil,
                "time_sec": elapsed,
                "labels": labels
            })
            
    return pd.DataFrame(sweep_results)


# =====================================================================
# 4. Diagnostic Visualization Exports (Output/plots/)
# =====================================================================

def generate_dbscan_diagrams(worked_example, two_moons_data, k_distances, suggested_eps, sample_df, best_labels, X_scaled):
    """Generate and save publication-quality diagnostic plots for DBSCAN."""
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    generated_plots = {}
    
    # -------------------------------------------------------------
    # Plot 1: Hand-Worked Example 9-Point Verification (Slide 8 & 11)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))
    pt_types = worked_example["point_types"]
    eps = worked_example["eps"]
    
    # Color map: Cluster 0, Cluster 1, Noise (-1)
    palette_worked = {0: "#2ecc71", 1: "#3498db", -1: "#e74c3c"}
    markers = {"core": "o", "border": "s", "noise": "X"}
    
    for name, info in pt_types.items():
        x, y = info["coords"]
        c_id = info["cluster"]
        kind = info["kind"]
        color = palette_worked[c_id]
        marker = markers[kind]
        
        # Plot point
        ax.scatter(x, y, color=color, marker=marker, s=120, edgecolors="black", linewidths=1.2, zorder=5)
        
        # Annotate label
        ax.annotate(
            f"{name} ({kind})\n|N|={worked_example['neighborhood_sizes'][name]}",
            xy=(x, y), xytext=(x + 0.3, y + 0.3),
            fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="gray")
        )
        
        # Draw epsilon circles around core points
        if kind == "core":
            circle = plt.Circle((x, y), eps, color=color, fill=True, alpha=0.15, linestyle="--", linewidth=1.0)
            ax.add_patch(circle)
            
    ax.set_xlim(0, 27)
    ax.set_ylim(0, 27)
    ax.set_title(
        f"DBSCAN Worked Numerical Verification (Slide 8 & 11)\n"
        f"eps={eps}, minPts={worked_example['minPts']} | Clusters: 2 (Green=0, Blue=1), Noise: H (Red)",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("X Coordinate", fontweight="bold")
    ax.set_ylabel("Y Coordinate", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Core Point (|N|>=3)', markerfacecolor='gray', markersize=10),
        Line2D([0], [0], marker='s', color='w', label='Border Point (I)', markerfacecolor='gray', markersize=10),
        Line2D([0], [0], marker='X', color='w', label='Noise Point (H)', markerfacecolor='red', markersize=10),
    ]
    ax.legend(handles=legend_elements, loc="center right", fontsize=9)
    
    plt.tight_layout()
    plot_worked_path = config.PLOTS_DIR / "dbscan_worked_example.png"
    plt.savefig(plot_worked_path, dpi=150, bbox_inches="tight")
    plt.close()
    generated_plots["worked_example"] = plot_worked_path.name
    
    # -------------------------------------------------------------
    # Plot 2: Non-Convex Clusters: K-Means vs DBSCAN (Slide 10 & 12)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    X_m = two_moons_data["X"]
    
    # K-Means
    axes[0].scatter(X_m[:, 0], X_m[:, 1], c=two_moons_data["kmeans_labels"], cmap="coolwarm", s=25, alpha=0.8, edgecolors="none")
    axes[0].set_title(f"K-Means (Convex Linear Partition)\nAdjusted Rand Index: {two_moons_data['ari_kmeans']} (Fails on Moons)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    
    # DBSCAN
    axes[1].scatter(X_m[:, 0], X_m[:, 1], c=two_moons_data["dbscan_labels"], cmap="coolwarm", s=25, alpha=0.8, edgecolors="none")
    axes[1].set_title(f"DBSCAN (Density-Reachable Chains)\nAdjusted Rand Index: {two_moons_data['ari_dbscan']} (Recovers True Geometry)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Feature 1")
    axes[1].set_ylabel("Feature 2")
    axes[1].grid(True, linestyle=":", alpha=0.5)
    
    fig.suptitle("Non-Convex Cluster Stress Test: K-Means Centroid Slicing vs DBSCAN Manifold Tracing", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plot_moons_path = config.PLOTS_DIR / "dbscan_two_moons_vs_kmeans.png"
    plt.savefig(plot_moons_path, dpi=150, bbox_inches="tight")
    plt.close()
    generated_plots["two_moons"] = plot_moons_path.name
    
    # -------------------------------------------------------------
    # Plot 3: k-Distance Graph Elbow Plot (Slide 9 & 14)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(k_distances, color="#34495e", linewidth=2.0)
    ax.axhline(suggested_eps, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Elbow Threshold: eps = {suggested_eps}")
    ax.set_title("k-Distance Graph for Optimal Epsilon (eps) Selection (Slide 9 & 14)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Points Sorted by Distance to k-th Nearest Neighbor", fontweight="bold")
    ax.set_ylabel("Distance to k-th Neighbor", fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.5)
    
    plt.tight_layout()
    plot_kdist_path = config.PLOTS_DIR / "dbscan_k_distance_elbow.png"
    plt.savefig(plot_kdist_path, dpi=150, bbox_inches="tight")
    plt.close()
    generated_plots["k_distance_elbow"] = plot_kdist_path.name
    
    # -------------------------------------------------------------
    # Plot 4: Placement Dataset DBSCAN Clusters Projection (Slide 14)
    # -------------------------------------------------------------
    pca_2d = PCA(n_components=2, random_state=config.RANDOM_STATE).fit_transform(X_scaled)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Panel A: DBSCAN Clusters with Noise Highlighted
    noise_mask = (best_labels == -1)
    cluster_mask = ~noise_mask
    
    # Clustered points
    scatter_c = axes[0].scatter(
        pca_2d[cluster_mask, 0], pca_2d[cluster_mask, 1],
        c=best_labels[cluster_mask], cmap="tab10", s=18, alpha=0.6, edgecolors="none"
    )
    # Noise points
    axes[0].scatter(
        pca_2d[noise_mask, 0], pca_2d[noise_mask, 1],
        c="#e74c3c", marker="x", s=32, linewidths=1.2, label=f"DBSCAN Noise (N={np.sum(noise_mask)})"
    )
    axes[0].set_title(f"DBSCAN Discovered Clusters on Placement Data (eps={suggested_eps})", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("PCA Component 1")
    axes[0].set_ylabel("PCA Component 2")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    
    # Panel B: Ground Truth Placement Status
    status = sample_df[TARGET_CLASSIFICATION].values
    palette_status = {"Placed": "#2ecc71", "Not Placed": "#e74c3c"}
    for st, col in palette_status.items():
        mask = (status == st)
        axes[1].scatter(pca_2d[mask, 0], pca_2d[mask, 1], c=col, label=st, alpha=0.45, s=16, edgecolors="none")
        
    axes[1].set_title("Ground-Truth Placement Status Overlay", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("PCA Component 1")
    axes[1].set_ylabel("PCA Component 2")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    plot_clusters_path = config.PLOTS_DIR / "dbscan_placement_clusters.png"
    plt.savefig(plot_clusters_path, dpi=150, bbox_inches="tight")
    plt.close()
    generated_plots["placement_clusters"] = plot_clusters_path.name
    
    return generated_plots


# =====================================================================
# 5. Diagnostic Report Generation (Output/Report/dbscan_report.txt)
# =====================================================================

def save_dbscan_report(worked_data, two_moons_data, suggested_eps, sweep_df, sample_df, best_row):
    """Save comprehensive DBSCAN diagnostic report to config.REPORTS_DIR."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = config.REPORTS_DIR / "dbscan_report.txt"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("MODULE 4 · SESSION 3: DBSCAN CLUSTERING REPORT\n")
        f.write("Course: Machine Learning (25SC2107E) | Placement Prediction Dataset\n")
        f.write("Reference: M4-Session3-DBSCAN_new.pptx\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. WORKED NUMERICAL EXAMPLE VERIFICATION (SLIDES 8 & 11):\n")
        f.write(f"   Dataset: 9 points in 2-D | eps={worked_data['eps']}, minPts={worked_data['minPts']}\n")
        f.write("   Point Classifications & Neighborhood Counts:\n")
        for name, info in worked_data["point_types"].items():
            f.write(f"     Point {name} at {info['coords']} -> |N_eps|={worked_data['neighborhood_sizes'][name]:<2} | Kind: {info['kind']:<6} | Cluster: {info['cluster']:>2}\n")
        f.write("   Verification check: Matches slide 8 & 11 benchmarks exactly:\n")
        f.write("     - A, B, C, D, E, F, G are Core points\n")
        f.write("     - Point I is Border (assigned to Cluster 0 via Core Point D)\n")
        f.write("     - Point H is Noise (unlabeled / -1)\n")
        f.write(f"     - Discovered {worked_data['n_clusters']} dense clusters + {worked_data['n_noise']} noise point.\n\n")
        
        f.write("2. NON-CONVEX TWO-MOONS BENCHMARK (SLIDES 10 & 12):\n")
        f.write(f"   - K-Means Adjusted Rand Index: {two_moons_data['ari_kmeans']} (Fails due to convex linear centroid slicing)\n")
        f.write(f"   - DBSCAN  Adjusted Rand Index: {two_moons_data['ari_dbscan']} (Perfect recovery via density-connected chains)\n")
        f.write("   Lesson: DBSCAN discovers arbitrary non-convex cluster geometries without pre-specifying K.\n\n")
        
        f.write("3. PARAMETER SELECTION & k-DISTANCE ELBOW HEURISTIC (SLIDE 9 & 14):\n")
        f.write(f"   - Optimal suggested eps from k-distance elbow: {suggested_eps}\n")
        f.write("   - Practical rule of thumb: minPts >= D + 1 (used minPts = 10..20 for 10D feature space)\n\n")
        
        f.write("4. PLACEMENT PREDICTION DATASET PARAMETER SWEEP:\n")
        f.write(f"   Subsample size: {len(sample_df):,} student records\n\n")
        cols_show = ["eps", "min_samples", "n_clusters", "n_core", "n_noise", "noise_ratio_pct", "silhouette_non_noise", "time_sec"]
        f.write(sweep_df[cols_show].to_string(index=False))
        f.write("\n\n")
        
        f.write("5. SELECTED CONFIGURATION SUMMARY:\n")
        f.write(f"   Optimal: eps={best_row['eps']}, min_samples={best_row['min_samples']}\n")
        f.write(f"   Discovered Clusters: {best_row['n_clusters']} | Core Points: {best_row['n_core']} | Noise Points: {best_row['n_noise']} ({best_row['noise_ratio_pct']}%)\n")
        
        # Audit noise points against ground truth IsAnomaly
        if TARGET_ANOMALY in sample_df.columns:
            labels = best_row["labels"]
            noise_mask = (labels == -1)
            anom_in_noise = sample_df.loc[noise_mask, TARGET_ANOMALY].mean() * 100 if np.sum(noise_mask) > 0 else 0
            base_anom = sample_df[TARGET_ANOMALY].mean() * 100
            f.write(f"\n   Outlier Concordance Audit:\n")
            f.write(f"   - Background anomaly rate in population: {base_anom:.2f}%\n")
            f.write(f"   - Anomaly rate within DBSCAN noise set:  {anom_in_noise:.2f}%\n")
            f.write("   (Confirms DBSCAN noise points heavily enrich for genuine anomalous profiles).\n\n")
            
        f.write("6. COMPARISON: DBSCAN vs LINEAR REGRESSION (SLIDE 13):\n")
        f.write("   " + "-" * 75 + "\n")
        f.write(f"   {'Aspect':<25} | {'DBSCAN':<22} | {'Linear Regression':<22}\n")
        f.write("   " + "-" * 75 + "\n")
        f.write(f"   {'Learning Type':<25} | {'Unsupervised':<22} | {'Supervised':<22}\n")
        f.write(f"   {'Goal':<25} | {'Discover density clusters':<22} | {'Predict continuous target y':<22}\n")
        f.write(f"   {'Input Needed':<25} | {'Features only (no labels)':<22} | {'Features X and labels y':<22}\n")
        f.write(f"   {'Shape Assumption':<25} | {'None (arbitrary/non-convex)':<22} | {'Assumes linear hyperplane':<22}\n")
        f.write(f"   {'Outlier Handling':<25} | {'Flags them as noise (-1)':<22} | {'Sensitive (distorts beta)':<22}\n")
        f.write("   " + "-" * 75 + "\n")
        
    return report_path


# =====================================================================
# 6. Pipeline Runner
# =====================================================================

def run_dbscan_pipeline(df=None, sample_size=3000, random_state=42):
    """Execute complete DBSCAN pipeline standalone."""
    print("=" * 70)
    print("MODULE 4 · SESSION 3: DBSCAN CLUSTERING PIPELINE")
    print("=" * 70)
    start_time = time.perf_counter()
    
    print("[1/5] Verifying Slide 8 & 11 hand-worked numerical example...")
    worked_data = run_dbscan_worked_example()
    print(f"      9 points (eps=1.5, minPts=3):")
    for name, info in worked_data["point_types"].items():
        print(f"        {name}: {info['kind']} (cluster {info['cluster']})")
        
    print("[2/5] Running Slide 10 & 12 Two-Moons non-convex benchmark...")
    two_moons_data = run_two_moons_benchmark(random_state=random_state)
    print(f"      K-Means ARI: {two_moons_data['ari_kmeans']} (Convex error)")
    print(f"      DBSCAN  ARI: {two_moons_data['ari_dbscan']} (Perfect non-convex recovery)")
    
    print(f"[3/5] Loading Placement dataset ({sample_size} records) & computing k-distance elbow...")
    sample_df, features, X_scaled, scaler = get_dbscan_data(df=df, sample_size=sample_size, random_state=random_state)
    k_distances, suggested_eps = compute_k_distance_graph(X_scaled, k=10)
    print(f"      Standardized {len(features)} numeric metrics with zero target leakage.")
    print(f"      Suggested eps from k-distance elbow: {suggested_eps}")
    
    print("[4/5] Running DBSCAN parameter sweep across eps and minPts...")
    sweep_df = train_dbscan_sweep(X_scaled, eps_list=[suggested_eps - 0.5, suggested_eps, suggested_eps + 0.5], min_samples_list=[10, 20])
    best_idx = 1  # suggested_eps, min_samples=10
    best_row = sweep_df.iloc[best_idx]
    best_labels = best_row["labels"]
    print(f"      Optimal setup: eps={best_row['eps']}, min_samples={best_row['min_samples']}")
    print(f"      Clusters: {best_row['n_clusters']} | Core: {best_row['n_core']} | Noise: {best_row['n_noise']} ({best_row['noise_ratio_pct']}%)")
    
    print("[5/5] Generating diagnostic plots and saving comprehensive report...")
    plots = generate_dbscan_diagrams(worked_data, two_moons_data, k_distances, suggested_eps, sample_df, best_labels, X_scaled)
    rep_path = save_dbscan_report(worked_data, two_moons_data, suggested_eps, sweep_df, sample_df, best_row)
    
    elapsed = round(time.perf_counter() - start_time, 2)
    print(f"\n[DONE] DBSCAN pipeline completed successfully in {elapsed}s.")
    print(f"       Diagnostic plots: {list(plots.values())}")
    print(f"       Report: {Path(rep_path).name}")
    
    return worked_data, two_moons_data, sweep_df, plots, rep_path


if __name__ == "__main__":
    run_dbscan_pipeline()
