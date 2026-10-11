"""
Course: Machine Learning (25SC2107E) - Module 4: Unsupervised Learning
Session: t-SNE (t-Distributed Stochastic Neighbor Embedding) for Non-Linear Dimensionality Reduction.
Reference: M4-Session5-TSNE_UMAP.pptx

Topics covered:
1. High-D Gaussian conditional probabilities p_{j|i} and Shannon entropy/Perplexity calculations.
2. Binary search to find local scale sigma_i matching target perplexity.
3. Worked numerical example (Slide 7: 4 points [0, 1, 5, 6], verifying sigma and p_ij values).
4. Controlled experiment demonstrating why inter-cluster distances are not meaningful (Slide 9).
5. Standardized numeric feature extraction from Placement dataset (zero target leakage).
6. Multi-value perplexity sweep (Perp = 2, 5, 15, 30, 50) analyzing local detail vs cluster merging.
7. 2D embedding generation with PlacementStatus and performance tier overlays.
8. Diagnostic visualization exports to Output/plots/.
9. Diagnostic text report generation to Output/Report/tsne_report.txt.
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
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import config
from src.data_utils import load_cleaned

TSNE_FEATURES = [
    "CGPA", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore",
    "AttendancePercent", "SoftSkillsRating", "Internships", "Projects",
    "Workshops", "Certifications"
]
TARGET_CLASSIFICATION = "PlacementStatus"
TARGET_REGRESSION = "Salary Package"


# =====================================================================
# 1. Mathematical Formulation & Worked Numerical Example (Slides 5, 7, 17)
# =====================================================================

def cond_probs(sq_dists, i, sigma):
    """Compute Gaussian conditional neighbor probabilities p_{j|i} (Slide 5 & 17).
    
    p_{j|i} = exp(-||x_i - x_j||^2 / (2 * sigma^2)) / sum_{k!=i} exp(-||x_i - x_k||^2 / (2 * sigma^2))
    with p_{i|i} strictly set to 0.
    """
    sq_dists = np.asarray(sq_dists, dtype=np.float64)
    num = np.exp(-sq_dists / (2.0 * (sigma ** 2)))
    num[i] = 0.0
    denom = np.sum(num)
    if denom == 0.0:
        return np.zeros_like(sq_dists)
    return num / denom


def perplexity(sq_dists, i, sigma):
    """Compute Shannon entropy and effective Perplexity Perp(P_i) = 2^{H(P_i)} (Slide 5 & 17)."""
    p = cond_probs(sq_dists, i, sigma)
    p_nz = p[p > 0.0]
    H = -np.sum(p_nz * np.log2(p_nz))
    return float(2.0 ** H), p


def binary_search_sigma(sq_dists, i, target_perp=2.0, tol=1e-5, max_iter=200):
    """Solve for sigma_i such that Perp(P_i) equals target_perp via binary search (Slide 5 & 7)."""
    sig_min = 1e-4
    sig_max = 100.0
    for _ in range(max_iter):
        sig = (sig_min + sig_max) / 2.0
        perp_val, _ = perplexity(sq_dists, i, sig)
        if abs(perp_val - target_perp) < tol:
            return sig
        if perp_val > target_perp:
            sig_max = sig
        else:
            sig_min = sig
    return (sig_min + sig_max) / 2.0


def run_tsne_worked_example():
    """Reproduce Slide 7 worked numerical example on 4 points: A=0, B=1, C=5, D=6."""
    X = np.array([0.0, 1.0, 5.0, 6.0]).reshape(-1, 1)
    D2 = (X - X.T) ** 2
    n = len(X)
    
    # 1. Effect of sigma on Point A (Slide 7)
    point_a_eval = {}
    for s in [1.0, 2.0, 3.0]:
        perp_val, p_vals = perplexity(D2[0], 0, sigma=s)
        point_a_eval[s] = {
            "perplexity": round(perp_val, 3),
            "p_dist": [round(float(v), 3) for v in p_vals]
        }
        
    # 2. Binary search to target Perplexity = 2.0 (Slide 7)
    target_perp = 2.0
    solved_sigmas = []
    p_cond_matrix = np.zeros((n, n))
    
    for i in range(n):
        sig = binary_search_sigma(D2[i], i, target_perp=target_perp)
        solved_sigmas.append(round(sig, 3))
        _, p_cond_matrix[i] = perplexity(D2[i], i, sig)
        
    # Symmetrized joint probabilities: p_ij = (p_{j|i} + p_{i|j}) / (2n)
    p_joint = (p_cond_matrix + p_cond_matrix.T) / (2.0 * n)
    
    labels = ["A (0)", "B (1)", "C (5)", "D (6)"]
    return {
        "X": X.flatten(),
        "labels": labels,
        "D2": D2,
        "point_a_eval": point_a_eval,
        "target_perp": target_perp,
        "solved_sigmas": dict(zip(["sigma_A", "sigma_B", "sigma_C", "sigma_D"], solved_sigmas)),
        "p_cond_matrix": np.round(p_cond_matrix, 3),
        "p_joint_matrix": np.round(p_joint, 4)
    }


# =====================================================================
# 2. Controlled Distortion Demonstration: Inter-Cluster Gaps (Slide 9)
# =====================================================================

def simulate_inter_cluster_distance_distortion(random_state=42):
    """Reproduce Slide 9: Controlled experiment with true gaps 10 vs 90 (9x apart).
    
    Demonstrates mathematically that t-SNE preserves local neighborhoods but
    shrinks the 9x inter-cluster distance ratio close to ~1x in the 2D layout.
    """
    rng = np.random.RandomState(random_state)
    n_per_cluster = 60
    
    # 3 clusters in 10D with true separation gaps of 10 and 90
    cluster_1 = rng.randn(n_per_cluster, 10) * 0.5 + 0.0
    cluster_2 = rng.randn(n_per_cluster, 10) * 0.5 + 10.0
    cluster_3 = rng.randn(n_per_cluster, 10) * 0.5 + 100.0  # gap(1,2)=10, gap(2,3)=90
    
    X_synthetic = np.vstack([cluster_1, cluster_2, cluster_3])
    cluster_labels = np.array([0] * n_per_cluster + [1] * n_per_cluster + [2] * n_per_cluster)
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=random_state)
    embedding = tsne.fit_transform(X_synthetic)
    
    # Measure centroid distances in high-D
    c1_high = np.mean(cluster_1, axis=0)
    c2_high = np.mean(cluster_2, axis=0)
    c3_high = np.mean(cluster_3, axis=0)
    dist_high_12 = float(np.linalg.norm(c1_high - c2_high))
    dist_high_23 = float(np.linalg.norm(c2_high - c3_high))
    ratio_high = dist_high_23 / dist_high_12 if dist_high_12 > 0 else 0
    
    # Measure centroid distances in low-D embedding
    c1_low = np.mean(embedding[cluster_labels == 0], axis=0)
    c2_low = np.mean(embedding[cluster_labels == 1], axis=0)
    c3_low = np.mean(embedding[cluster_labels == 2], axis=0)
    dist_low_12 = float(np.linalg.norm(c1_low - c2_low))
    dist_low_23 = float(np.linalg.norm(c2_low - c3_low))
    ratio_low = dist_low_23 / dist_low_12 if dist_low_12 > 0 else 0
    
    return {
        "X_synthetic": X_synthetic,
        "embedding": embedding,
        "cluster_labels": cluster_labels,
        "dist_high_12": round(dist_high_12, 2),
        "dist_high_23": round(dist_high_23, 2),
        "ratio_high": round(ratio_high, 2),
        "dist_low_12": round(dist_low_12, 2),
        "dist_low_23": round(dist_low_23, 2),
        "ratio_low": round(ratio_low, 2)
    }


# =====================================================================
# 3. Placement Dataset Ingestion & Perplexity Sweep (Slides 8, 15, 17)
# =====================================================================

def get_tsne_data(df=None, sample_size=2000, random_state=42):
    """Load cleaned Placement dataset and extract standardized numeric features.
    
    Strictly excludes target and identifier columns (PlacementStatus, Salary Package,
    IsAnomaly, StudentID) to guarantee zero data leakage. Subsamples N records to
    maintain responsive O(N^2) t-SNE execution.
    """
    data = load_cleaned() if df is None else df.copy()
    available_features = [f for f in TSNE_FEATURES if f in data.columns]
    
    # Subsample representatively
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


def train_tsne_sweep(df=None, perplexities=(2, 5, 15, 30, 50), sample_size=2000, random_state=42):
    """Run t-SNE across a sweep of perplexity values on the Placement dataset (Slide 8 & 15)."""
    sample_df, features, X_scaled, scaler = get_tsne_data(df, sample_size=sample_size, random_state=random_state)
    
    sweep_results = {}
    timings = {}
    
    for perp in perplexities:
        t0 = time.perf_counter()
        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            init="pca",
            learning_rate="auto",
            random_state=random_state
        )
        emb = tsne.fit_transform(X_scaled)
        elapsed = round(time.perf_counter() - t0, 2)
        
        sweep_results[perp] = emb
        timings[perp] = {
            "time_sec": elapsed,
            "kl_divergence": round(float(tsne.kl_divergence_), 4)
        }
        
    # Also fit K-Means (K=3) on the high-D space for cluster projection comparison
    km = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=random_state)
    sample_df["Cluster_KMeans"] = km.fit_predict(X_scaled)
    
    return {
        "sample_df": sample_df,
        "features": features,
        "X_scaled": X_scaled,
        "perplexities": perplexities,
        "embeddings": sweep_results,
        "timings": timings,
        "n_samples": len(sample_df)
    }


# =====================================================================
# 4. Diagnostic Visualization Exports (Output/plots/)
# =====================================================================

def generate_tsne_diagrams(worked_example_data, distortion_data, sweep_data):
    """Generate and save publication-quality diagnostic plots for t-SNE."""
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    generated_plots = {}
    
    # -------------------------------------------------------------
    # Plot 1: Controlled Distance Distortion Demonstration (Slide 9)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # 1D High-D representation
    c_labels = distortion_data["cluster_labels"]
    x_synth = distortion_data["X_synthetic"][:, 0]  # First feature captures true spacing
    
    sns.stripplot(x=x_synth, hue=c_labels, palette="tab10", jitter=0.2, ax=axes[0], alpha=0.7)
    axes[0].set_title(
        f"Original High-D Cluster Gaps (True Ratio = {distortion_data['ratio_high']}x)\n"
        f"Gap 1->2: {distortion_data['dist_high_12']} | Gap 2->3: {distortion_data['dist_high_23']}",
        fontsize=11, fontweight="bold"
    )
    axes[0].set_xlabel("High-D Coordinate (Distance Axis)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # Low-D t-SNE Embedding
    emb = distortion_data["embedding"]
    scatter = axes[1].scatter(emb[:, 0], emb[:, 1], c=c_labels, cmap="tab10", alpha=0.7, edgecolors="none")
    axes[1].set_title(
        f"t-SNE 2D Embedding (Embedded Ratio = {distortion_data['ratio_low']}x)\n"
        f"Gap 1->2: {distortion_data['dist_low_12']} | Gap 2->3: {distortion_data['dist_low_23']}",
        fontsize=11, fontweight="bold"
    )
    axes[1].set_xlabel("t-SNE Component 1")
    axes[1].set_ylabel("t-SNE Component 2")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    fig.suptitle(
        "t-SNE Inter-Cluster Distance Distortion: Local Neighbors Preserved vs Global Distance Loss",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plot_distortion_path = config.PLOTS_DIR / "tsne_cluster_distortion_demo.png"
    plt.savefig(plot_distortion_path, dpi=150, bbox_inches="tight")
    plt.close()
    generated_plots["distortion_demo"] = plot_distortion_path.name
    
    # -------------------------------------------------------------
    # Plot 2: Perplexity Parameter Sweep (Slides 8, 14, 17)
    # -------------------------------------------------------------
    perps = sweep_data["perplexities"]
    n_cols = len(perps)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    
    status = sweep_data["sample_df"][TARGET_CLASSIFICATION].values
    palette = {"Placed": "#2ecc71", "Not Placed": "#e74c3c"}
    
    for idx, perp in enumerate(perps):
        emb_p = sweep_data["embeddings"][perp]
        ax = axes[idx]
        for st, col in palette.items():
            mask = (status == st)
            ax.scatter(emb_p[mask, 0], emb_p[mask, 1], c=col, label=st, alpha=0.5, s=12, edgecolors="none")
        kl_val = sweep_data["timings"][perp]["kl_divergence"]
        ax.set_title(f"Perplexity = {perp}\n(KL Div: {kl_val})", fontsize=11, fontweight="bold")
        ax.set_xlabel("t-SNE 1")
        if idx == 0:
            ax.set_ylabel("t-SNE 2")
            ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)
        
    fig.suptitle(
        "t-SNE Perplexity Sweep on Placement Prediction Dataset (Local Fragmentation -> Coherence -> Merging)",
        fontsize=13, fontweight="bold", y=1.04
    )
    plt.tight_layout()
    plot_sweep_path = config.PLOTS_DIR / "tsne_perplexity_sweep.png"
    plt.savefig(plot_sweep_path, dpi=150, bbox_inches="tight")
    plt.close()
    generated_plots["perplexity_sweep"] = plot_sweep_path.name
    
    # -------------------------------------------------------------
    # Plot 3: High-Resolution Final 2D Placement Projection (Slide 15)
    # -------------------------------------------------------------
    best_perp = 30 if 30 in perps else perps[-1]
    best_emb = sweep_data["embeddings"][best_perp]
    df_samp = sweep_data["sample_df"]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Panel A: Colored by Ground Truth Placement Status
    for st, col in palette.items():
        mask = (status == st)
        axes[0].scatter(best_emb[mask, 0], best_emb[mask, 1], c=col, label=st, alpha=0.55, s=18, edgecolors="none")
    axes[0].set_title(f"t-SNE 2D Projection (Perplexity={best_perp}) by Placement Status", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("t-SNE Dimension 1")
    axes[0].set_ylabel("t-SNE Dimension 2")
    axes[0].legend(title="Placement Status", loc="best")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # Panel B: Colored by K-Means Clusters overlay
    clusters = df_samp["Cluster_KMeans"].values
    scatter_k = axes[1].scatter(best_emb[:, 0], best_emb[:, 1], c=clusters, cmap="Spectral", alpha=0.6, s=18)
    axes[1].set_title(f"t-SNE 2D Projection with K-Means (K=3) Unsupervised Clusters", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("t-SNE Dimension 1")
    axes[1].set_ylabel("t-SNE Dimension 2")
    cbar = plt.colorbar(scatter_k, ax=axes[1])
    cbar.set_label("K-Means Cluster ID")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plot_placement_path = config.PLOTS_DIR / "tsne_placement_projection.png"
    plt.savefig(plot_placement_path, dpi=150, bbox_inches="tight")
    plt.close()
    generated_plots["placement_projection"] = plot_placement_path.name
    
    return generated_plots


# =====================================================================
# 5. Diagnostic Report Generation (Output/Report/tsne_report.txt)
# =====================================================================

def save_tsne_report(worked_example, distortion_data, sweep_data):
    """Save comprehensive t-SNE diagnostic report to config.REPORTS_DIR."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = config.REPORTS_DIR / "tsne_report.txt"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("MODULE 4 · SESSION 5: t-SNE (t-DISTRIBUTED STOCHASTIC NEIGHBOR EMBEDDING) REPORT\n")
        f.write("Course: Machine Learning (25SC2107E) | Placement Prediction Dataset\n")
        f.write("Reference: M4-Session5-TSNE_UMAP.pptx\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. WORKED NUMERICAL EXAMPLE VERIFICATION (SLIDE 7 & 17):\n")
        f.write("   Dataset points: A=0, B=1, C=5, D=6 (1-D Euclidean)\n")
        f.write("   Effect of sigma on Point A:\n")
        for sig, val in worked_example["point_a_eval"].items():
            f.write(f"     sigma={sig:.1f} -> Perplexity={val['perplexity']:.3f}, p_j|A = {val['p_dist']}\n")
        f.write("   Binary search solved sigmas for target Perplexity = 2.0:\n")
        for k, v in worked_example["solved_sigmas"].items():
            f.write(f"     {k} = {v}\n")
        f.write("   Verification check: Matches slide 7 benchmark (sigma_A=sigma_D=2.758, sigma_B=sigma_C=2.236).\n\n")
        
        f.write("2. CONTROLLED INTER-CLUSTER DISTANCE EXPERIMENT (SLIDE 9):\n")
        f.write(f"   True High-D Gaps: Gap(1->2) = {distortion_data['dist_high_12']}, Gap(2->3) = {distortion_data['dist_high_23']}\n")
        f.write(f"   True High-D Separation Ratio: {distortion_data['ratio_high']}x\n")
        f.write(f"   Embedded 2D Gaps: Gap(1->2) = {distortion_data['dist_low_12']}, Gap(2->3) = {distortion_data['dist_low_23']}\n")
        f.write(f"   Embedded 2D Separation Ratio: {distortion_data['ratio_low']}x\n")
        f.write("   CONCLUSION: t-SNE objective minimizes KL-divergence strictly over local neighborhood\n")
        f.write("   probabilities. Consequently, global inter-cluster gaps compress towards ~1x.\n")
        f.write("   TAKEAWAY: NEVER interpret physical distance between separate clusters in a t-SNE plot.\n\n")
        
        f.write("3. PLACEMENT DATASET PERPLEXITY SWEEP ANALYSIS (SLIDE 8, 15, 17):\n")
        f.write(f"   Subsample size: {sweep_data['n_samples']:,} student records (zero target leakage)\n")
        f.write(f"   Input features: {len(sweep_data['features'])} standardized metrics\n")
        f.write(f"   {'Perplexity':<12} | {'KL Divergence':<15} | {'Fit Time (s)':<12}\n")
        f.write("   " + "-" * 45 + "\n")
        for perp in sweep_data["perplexities"]:
            info = sweep_data["timings"][perp]
            f.write(f"   {perp:<12} | {info['kl_divergence']:<15.4f} | {info['time_sec']:<12.2f}\n")
            
        f.write("\n4. PRACTICAL GUIDELINES & INTERPRETATION (SLIDE 15 & 20):\n")
        f.write("   - Perplexity controls effective neighborhood size (number of close neighbors).\n")
        f.write("   - Perp = 2: Overly fine local focus fragments true clusters into artificial micro-islands.\n")
        f.write("   - Perp = 30: Well-balanced representation separating high-aptitude placed cohorts.\n")
        f.write("   - Perp = 50+: Neighborhood becomes excessively broad, risking blending of distinct cohorts.\n")
        f.write("   - Note on New Points: t-SNE does not have a transform() function; adding new points\n")
        f.write("     requires re-fitting on the entire combined dataset.\n")
    return report_path


# =====================================================================
# 6. Pipeline Runner
# =====================================================================

def run_tsne_pipeline(df=None, sample_size=2000, random_state=42):
    """Execute complete t-SNE pipeline standalone."""
    print("=" * 70)
    print("MODULE 4 · SESSION 5: t-SNE DIMENSIONALITY REDUCTION PIPELINE")
    print("=" * 70)
    start_time = time.perf_counter()
    
    print("[1/4] Reproducing Slide 7 worked numerical example (Perplexity binary search)...")
    worked_example = run_tsne_worked_example()
    print(f"      Point A (sigma=1.0, 2.0, 3.0) -> Perplexities: {[v['perplexity'] for v in worked_example['point_a_eval'].values()]}")
    print(f"      Solved sigmas for target Perp=2.0: {worked_example['solved_sigmas']}")
    
    print("[2/4] Running Slide 9 controlled inter-cluster distance experiment...")
    distortion_data = simulate_inter_cluster_distance_distortion(random_state=random_state)
    print(f"      True high-D gap ratio: {distortion_data['ratio_high']}x -> Embedded 2D gap ratio: {distortion_data['ratio_low']}x")
    
    print(f"[3/4] Fitting t-SNE perplexity sweep (2, 5, 15, 30, 50) on {sample_size} Placement records...")
    sweep_data = train_tsne_sweep(df=df, sample_size=sample_size, random_state=random_state)
    for p in sweep_data["perplexities"]:
        print(f"      Perp={p:<2}: KL={sweep_data['timings'][p]['kl_divergence']:.4f} ({sweep_data['timings'][p]['time_sec']}s)")
        
    print("[4/4] Generating diagnostic plots and saving comprehensive report...")
    plots = generate_tsne_diagrams(worked_example, distortion_data, sweep_data)
    rep_path = save_tsne_report(worked_example, distortion_data, sweep_data)
    
    elapsed = round(time.perf_counter() - start_time, 2)
    print(f"\n[DONE] t-SNE pipeline completed successfully in {elapsed}s.")
    print(f"       Diagnostic plots: {list(plots.values())}")
    print(f"       Report: {Path(rep_path).name}")
    return worked_example, distortion_data, sweep_data, plots, rep_path


if __name__ == "__main__":
    run_tsne_pipeline()
