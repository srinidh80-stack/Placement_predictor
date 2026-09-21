"""
Course: Machine Learning (25SC2107E) - Module 4: Unsupervised Learning
Session 21: K-Means Clustering on Placement Prediction Dataset.

Topics covered:
1. Feature extraction & Z-score scaling (zero-leakage: targets strictly excluded).
2. K-Means sweep across K = 2..10 using k-means++ initialization.
3. Inertia (Within-Cluster Sum of Squares) calculation for the Elbow method.
4. Silhouette Coefficient analysis on representative subsample.
5. Optimal K determination via silhouette peak and elbow curvature.
6. Post-hoc placement rate and academic/coding profile analysis per cluster.
7. Diagnostic plot exports (Elbow & Silhouette curves, 2D PCA cluster projection).
8. Text report generation (kmeans_report.txt and kmeans_k_selection_results.csv).
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
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import config
from src.data_utils import load_cleaned

KMEANS_FEATURES = [
    "CGPA", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore",
    "AttendancePercent", "SoftSkillsRating", "Internships", "Projects",
    "Workshops", "Certifications"
]
TARGET_CLASSIFICATION = "PlacementStatus"
TARGET_REGRESSION = "Salary Package"


def get_kmeans_data(df=None):
    """Load cleaned dataset and extract standardized numeric features for K-Means.
    
    Excludes all target and identifier columns (PlacementStatus, Salary Package,
    IsAnomaly, StudentID) to guarantee zero data leakage.
    """
    data = load_cleaned() if df is None else df.copy()
    available_features = [f for f in KMEANS_FEATURES if f in data.columns]
    
    X_raw = data[available_features].copy()
    for col in available_features:
        X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce")
        
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X_raw), columns=available_features, index=X_raw.index
    )
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    return data, available_features, X_imputed, X_scaled, imputer, scaler


def train_kmeans_sweep(df=None, k_min=2, k_max=10, silhouette_samples=5000, random_state=42):
    """Fit K-Means across K=2..10, record Inertia and Silhouette, and select optimal K."""
    data, features, X_imputed, X_scaled, imputer, scaler = get_kmeans_data(df)
    n_samples = len(X_scaled)
    
    k_range = list(range(k_min, k_max + 1))
    sweep_results = []
    
    # Subsample for silhouette to keep O(n^2) distance computation fast and responsive
    rng = np.random.RandomState(random_state)
    sample_size = min(silhouette_samples, n_samples)
    sample_idx = rng.choice(n_samples, size=sample_size, replace=False)
    X_sample = X_scaled[sample_idx]
    
    models = {}
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
        labels = km.fit_predict(X_scaled)
        inertia = float(km.inertia_)
        sil = float(silhouette_score(X_sample, labels[sample_idx]))
        
        sweep_results.append({
            "K": k,
            "Inertia": round(inertia, 2),
            "Silhouette": round(sil, 4)
        })
        models[k] = km
        
    results_df = pd.DataFrame(sweep_results)
    
    # Best K by maximum silhouette score
    best_k_silhouette = int(results_df.loc[results_df["Silhouette"].idxmax(), "K"])
    best_silhouette_val = float(results_df["Silhouette"].max())
    
    # Numerical elbow proxy via maximum second difference of inertia
    inertia_vals = results_df["Inertia"].values
    if len(inertia_vals) >= 3:
        first_diff = np.diff(inertia_vals)
        second_diff = np.diff(first_diff)
        elbow_k = int(results_df["K"].values[int(np.argmax(second_diff)) + 1])
    else:
        elbow_k = best_k_silhouette
        
    optimal_k = best_k_silhouette
    final_km = models[optimal_k]
    cluster_labels = final_km.labels_
    
    data_with_clusters = data.copy()
    data_with_clusters["KMeans_Cluster"] = cluster_labels
    
    # Discovered Cluster Profiling
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    cluster_profiles = []
    
    for c_id in range(optimal_k):
        c_mask = (cluster_labels == c_id)
        c_size = int(cluster_counts[c_id])
        c_pct = round(c_size / n_samples * 100, 1)
        
        profile = {
            "cluster_id": c_id,
            "size": c_size,
            "percentage": c_pct,
        }
        
        for feat in ["CGPA", "CodingTestScore", "AptitudeTestScore", "SoftSkillsRating", "Internships", "Projects"]:
            if feat in data.columns:
                profile[feat] = round(float(pd.to_numeric(data.loc[c_mask, feat], errors="coerce").mean()), 2)
                
        # Unsupervised cohort placement rate validation
        if TARGET_CLASSIFICATION in data.columns:
            target_series = pd.to_numeric(data.loc[c_mask, TARGET_CLASSIFICATION], errors="coerce").dropna()
            profile["placement_rate"] = round(float(target_series.mean() * 100), 1) if len(target_series) > 0 else 0.0
        else:
            profile["placement_rate"] = 0.0
            
        cluster_profiles.append(profile)
        
    cluster_profiles_sorted = sorted(cluster_profiles, key=lambda x: x["placement_rate"], reverse=True)
    role_names = ["High Achiever Cohort", "Core / Mid-Tier Cohort", "Developing / At-Risk Cohort"]
    for i, prof in enumerate(cluster_profiles_sorted):
        prof["cohort_label"] = role_names[i] if i < len(role_names) else f"Cohort {prof['cluster_id'] + 1}"
        
    return {
        "features": features,
        "n_samples": n_samples,
        "k_range": k_range,
        "sweep_df": results_df,
        "optimal_k": optimal_k,
        "best_silhouette": best_silhouette_val,
        "elbow_k": elbow_k,
        "final_model": final_km,
        "imputer": imputer,
        "scaler": scaler,
        "X_scaled": X_scaled,
        "cluster_labels": cluster_labels,
        "cluster_profiles": cluster_profiles,
        "data_with_clusters": data_with_clusters
    }


def generate_kmeans_diagrams(kmeans_data):
    """Generate Elbow & Silhouette curve and 2D Cluster Projection."""
    results_df = kmeans_data["sweep_df"]
    optimal_k = kmeans_data["optimal_k"]
    elbow_k = kmeans_data["elbow_k"]
    X_scaled = kmeans_data["X_scaled"]
    labels = kmeans_data["cluster_labels"]
    
    # 1. Dual Elbow and Silhouette Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Elbow (Inertia)
    axes[0].plot(results_df["K"], results_df["Inertia"], marker="o", color="#0f766e", linewidth=2.5, markersize=8)
    axes[0].axvline(x=elbow_k, color="#f59e0b", linestyle="--", linewidth=1.8, label=f"Elbow Point (K={elbow_k})")
    axes[0].set_title("Elbow Method: Inertia (WCSS) vs K", fontsize=13, fontweight="bold", pad=10)
    axes[0].set_xlabel("Number of Clusters (K)", fontsize=11)
    axes[0].set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=11)
    axes[0].set_xticks(results_df["K"].tolist())
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=10)
    
    # Silhouette
    axes[1].plot(results_df["K"], results_df["Silhouette"], marker="s", color="#8b5cf6", linewidth=2.5, markersize=8)
    axes[1].axvline(x=optimal_k, color="#10b981", linestyle="--", linewidth=1.8, label=f"Optimal Silhouette (K={optimal_k})")
    axes[1].set_title("Silhouette Analysis: Quality vs K", fontsize=13, fontweight="bold", pad=10)
    axes[1].set_xlabel("Number of Clusters (K)", fontsize=11)
    axes[1].set_ylabel("Mean Silhouette Coefficient", fontsize=11)
    axes[1].set_xticks(results_df["K"].tolist())
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=10)
    
    plt.suptitle("K-Means Cluster Selection Diagnostics (Module 4 · Session 21)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    elbow_sil_path = config.PLOTS_DIR / "kmeans_elbow_and_silhouette.png"
    plt.savefig(elbow_sil_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    # 2. 2D PCA Cluster Projection
    pca = PCA(n_components=2, random_state=42)
    plot_n = min(3000, len(X_scaled))
    rng = np.random.RandomState(42)
    sub_idx = rng.choice(len(X_scaled), size=plot_n, replace=False)
    
    X_pca = pca.fit_transform(X_scaled[sub_idx])
    sub_labels = labels[sub_idx]
    
    centroids_scaled = kmeans_data["final_model"].cluster_centers_
    centroids_pca = pca.transform(centroids_scaled)
    
    plt.figure(figsize=(10, 6.5))
    palette = sns.color_palette("bright", optimal_k)
    
    for c_id in range(optimal_k):
        mask = (sub_labels == c_id)
        plt.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=[palette[c_id]], alpha=0.45, s=28,
            label=f"Cluster {c_id} ({kmeans_data['cluster_profiles'][c_id]['cohort_label']})"
        )
        
    plt.scatter(
        centroids_pca[:, 0], centroids_pca[:, 1],
        marker="X", s=250, c="#dc2626", edgecolor="white", linewidth=2,
        label="Cluster Centroids"
    )
    
    var_exp = pca.explained_variance_ratio_
    plt.title(f"2D PCA Projection of K-Means Clusters (K={optimal_k})\nExplained Variance: PC1={var_exp[0]*100:.1f}%, PC2={var_exp[1]*100:.1f}%",
              fontsize=13, fontweight="bold", pad=12)
    plt.xlabel(f"Principal Component 1 ({var_exp[0]*100:.1f}% var)", fontsize=11)
    plt.ylabel(f"Principal Component 2 ({var_exp[1]*100:.1f}% var)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    scatter_path = config.PLOTS_DIR / "kmeans_cluster_scatter.png"
    plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    return {
        "plot_elbow_silhouette": "kmeans_elbow_and_silhouette.png",
        "plot_cluster_scatter": "kmeans_cluster_scatter.png"
    }


def save_kmeans_report(kmeans_data):
    """Save K-Means diagnostic report and parameter sweep table."""
    results_df = kmeans_data["sweep_df"]
    results_df.to_csv(config.REPORTS_DIR / "kmeans_k_selection_results.csv", index=False)
    
    km_report_path = config.REPORTS_DIR / "kmeans_report.txt"
    with open(km_report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("MODULE 4 · SESSION 21: K-MEANS CLUSTERING REPORT\n")
        f.write("Course: Machine Learning (25SC2107E) | Placement Prediction Dataset\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Dataset Size: {kmeans_data['n_samples']:,} student records\n")
        f.write(f"Feature Space ({len(kmeans_data['features'])} numeric metrics):\n")
        for ft in kmeans_data["features"]:
            f.write(f"  - {ft}\n")
        f.write("\nNOTE ON ZERO LEAKAGE:\n")
        f.write("Target variables (PlacementStatus, Salary Package, IsAnomaly, StudentID)\n")
        f.write("were strictly excluded from clustering to ensure completely unsupervised groupings.\n\n")
        f.write("K-MEANS HYPERPARAMETER SWEEP (K=2..10):\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\nOPTIMAL CLUSTER SELECTION:\n")
        f.write(f"  - Best K by Max Silhouette Score: K = {kmeans_data['optimal_k']} (Silhouette = {kmeans_data['best_silhouette']:.4f})\n")
        f.write(f"  - Elbow Point (Max Second Difference): K = {kmeans_data['elbow_k']}\n\n")
        f.write(f"DISCOVERED CLUSTER PROFILES (K={kmeans_data['optimal_k']}):\n")
        for prof in kmeans_data["cluster_profiles"]:
            f.write(f"\n  Cluster {prof['cluster_id']} [{prof['cohort_label']}]:\n")
            f.write(f"    - Student Count: {prof['size']:,} ({prof['percentage']}% of cohort)\n")
            f.write(f"    - Average CGPA: {prof.get('CGPA', 'N/A')}\n")
            f.write(f"    - Average Coding Score: {prof.get('CodingTestScore', 'N/A')}\n")
            f.write(f"    - Average Aptitude Score: {prof.get('AptitudeTestScore', 'N/A')}\n")
            f.write(f"    - Average Internships: {prof.get('Internships', 'N/A')}\n")
            f.write(f"    - Post-Hoc Placement Rate: {prof['placement_rate']}% (discovered naturally!)\n")


def run_kmeans_pipeline():
    """Run Session 21 pipeline standalone."""
    print("=" * 60)
    print("MODULE 4 · SESSION 21: K-MEANS CLUSTERING")
    print("=" * 60)
    start_time = time.time()
    
    print("[1/3] Sweeping K-Means across K=2..10...")
    km_data = train_kmeans_sweep()
    print(f"      Optimal K by Silhouette: K={km_data['optimal_k']} (Silhouette={km_data['best_silhouette']:.4f})")
    print(f"      Elbow Point             : K={km_data['elbow_k']}")
    
    print("[2/3] Generating diagnostic curves and PCA projection...")
    km_plots = generate_kmeans_diagrams(km_data)
    print(f"      Saved: {km_plots['plot_elbow_silhouette']}, {km_plots['plot_cluster_scatter']}")
    
    print("[3/3] Saving executive report and parameter table...")
    save_kmeans_report(km_data)
    print(f"      Saved: kmeans_report.txt, kmeans_k_selection_results.csv")
    
    elapsed = round(time.time() - start_time, 2)
    print(f"[DONE] K-Means completed successfully in {elapsed}s.")
    return km_data, km_plots


if __name__ == "__main__":
    run_kmeans_pipeline()
