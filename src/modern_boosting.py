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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

import config
from src.data_utils import load_cleaned

MODERN_FEATURES = [
    "CGPA", "CodingTestScore", "MockInterviewScore", "AptitudeTestScore",
    "AttendancePercent", "SoftSkillsRating", "Internships", "Projects"
]
TARGET_CLASSIFICATION = "PlacementStatus"


def check_library_availability():
    """Verify which modern boosted tree libraries are installed."""
    available = {}
    try:
        import xgboost
        available["xgboost"] = True
    except ImportError:
        available["xgboost"] = False
        
    try:
        import lightgbm
        available["lightgbm"] = True
    except ImportError:
        available["lightgbm"] = False
        
    try:
        import catboost
        available["catboost"] = True
    except ImportError:
        available["catboost"] = False
        
    return available


def train_modern_boosting_benchmarks(df=None, test_size=0.2, random_state=42):
    """Train XGBoost, LightGBM, CatBoost (and HistGradientBoosting) and benchmark time & accuracy."""
    data = load_cleaned() if df is None else df.copy()
    features = [f for f in MODERN_FEATURES if f in data.columns]
    
    X = data[features].fillna(data[features].median())
    y = data[TARGET_CLASSIFICATION].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    avail = check_library_availability()
    models = {}
    benchmarks = []
    
    # 1. XGBoost
    if avail["xgboost"]:
        import xgboost as xgb
        clf_xgb = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=random_state,
            eval_metric="logloss",
            verbosity=0
        )
        t0 = time.perf_counter()
        clf_xgb.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        pred_xgb = clf_xgb.predict(X_test)
        prob_xgb = clf_xgb.predict_proba(X_test)[:, 1]
        infer_time = (time.perf_counter() - t0) * 1000 / len(X_test)
        
        models["xgboost"] = clf_xgb
        benchmarks.append({
            "model": "XGBoost",
            "framework": "Extreme Gradient Boosting",
            "train_time_sec": round(fit_time, 3),
            "infer_latency_ms": round(infer_time, 3),
            "accuracy": round(float(accuracy_score(y_test, pred_xgb) * 100), 2),
            "f1": round(float(f1_score(y_test, pred_xgb, zero_division=0) * 100), 2),
            "roc_auc": round(float(roc_auc_score(y_test, prob_xgb) * 100), 2),
            "key_strength": "Level-wise growth, exact split finding, robust regularized objective (L1/L2)",
        })

    # 2. LightGBM
    if avail["lightgbm"]:
        import lightgbm as lgb
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            random_state=random_state,
            verbosity=-1
        )
        t0 = time.perf_counter()
        clf_lgb.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        pred_lgb = clf_lgb.predict(X_test)
        prob_lgb = clf_lgb.predict_proba(X_test)[:, 1]
        infer_time = (time.perf_counter() - t0) * 1000 / len(X_test)
        
        models["lightgbm"] = clf_lgb
        benchmarks.append({
            "model": "LightGBM",
            "framework": "Light Gradient Boosted Machine",
            "train_time_sec": round(fit_time, 3),
            "infer_latency_ms": round(infer_time, 3),
            "accuracy": round(float(accuracy_score(y_test, pred_lgb) * 100), 2),
            "f1": round(float(f1_score(y_test, pred_lgb, zero_division=0) * 100), 2),
            "roc_auc": round(float(roc_auc_score(y_test, prob_lgb) * 100), 2),
            "key_strength": "Leaf-wise (best-first) growth, GOSS (Gradient-based One-Side Sampling), ultra fast training",
        })

    # 3. CatBoost
    if avail["catboost"]:
        import catboost as cb
        clf_cb = cb.CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=6,
            random_seed=random_state,
            verbose=False
        )
        t0 = time.perf_counter()
        clf_cb.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        pred_cb = clf_cb.predict(X_test)
        prob_cb = clf_cb.predict_proba(X_test)[:, 1]
        infer_time = (time.perf_counter() - t0) * 1000 / len(X_test)
        
        models["catboost"] = clf_cb
        benchmarks.append({
            "model": "CatBoost",
            "framework": "Categorical Boosting",
            "train_time_sec": round(fit_time, 3),
            "infer_latency_ms": round(infer_time, 3),
            "accuracy": round(float(accuracy_score(y_test, pred_cb) * 100), 2),
            "f1": round(float(f1_score(y_test, pred_cb, zero_division=0) * 100), 2),
            "roc_auc": round(float(roc_auc_score(y_test, prob_cb) * 100), 2),
            "key_strength": "Symmetric (oblivious) trees, ordered boosting, native high-performance categorical handling",
        })

    # 4. Built-in HistGradientBoosting (Baseline modern benchmark always available in scikit-learn)
    clf_hgb = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=random_state)
    t0 = time.perf_counter()
    clf_hgb.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    pred_hgb = clf_hgb.predict(X_test)
    prob_hgb = clf_hgb.predict_proba(X_test)[:, 1]
    infer_time = (time.perf_counter() - t0) * 1000 / len(X_test)
    
    models["hist_gb"] = clf_hgb
    benchmarks.append({
        "model": "Scikit-Learn HistGB",
        "framework": "Histogram-Based Gradient Boosting",
        "train_time_sec": round(fit_time, 3),
        "infer_latency_ms": round(infer_time, 3),
        "accuracy": round(float(accuracy_score(y_test, pred_hgb) * 100), 2),
        "f1": round(float(f1_score(y_test, pred_hgb, zero_division=0) * 100), 2),
        "roc_auc": round(float(roc_auc_score(y_test, prob_hgb) * 100), 2),
        "key_strength": "Integrated in sklearn, C-optimized integer binning, zero external dependencies",
    })

    return {
        "models": models,
        "benchmarks": benchmarks,
        "features": features,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "available_libraries": avail
    }


def generate_modern_boosting_diagrams(model_data=None, df=None):
    """Generate Comparative Benchmark Chart and XGBoost Feature Importance Plot."""
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if model_data is None:
        model_data = train_modern_boosting_benchmarks(df=df)
        
    benchmarks = model_data["benchmarks"]
    models = model_data["models"]
    features = model_data["features"]
    
    sns.set_theme(style="whitegrid")
    
    # -------------------------------------------------------------
    # DIAGRAM 1: Benchmark Comparison (Accuracy vs Training Speed)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=120)
    
    model_names = [b["model"] for b in benchmarks]
    accs = [b["accuracy"] for b in benchmarks]
    times = [b["train_time_sec"] for b in benchmarks]
    
    colors = ["#0f9f8f", "#3a6b88", "#f5b84b", "#e05242"][:len(model_names)]
    
    # Subplot A: Accuracy
    ax1.bar(model_names, accs, color=colors, edgecolor="#21312f", linewidth=0.8, alpha=0.9)
    ax1.set_title("Test Accuracy Comparison", fontsize=10.5, weight="bold", color="#21312f")
    ax1.set_ylabel("Accuracy (%)", fontsize=9, weight="bold", color="#21312f")
    min_acc = min(accs) - 2.0 if accs else 50.0
    ax1.set_ylim(max(0, min_acc), 100)
    for i, v in enumerate(accs):
        ax1.text(i, v + 0.4, f"{v}%", ha="center", fontsize=8.5, weight="bold")
    ax1.tick_params(axis="x", rotation=15)
    
    # Subplot B: Training Time
    ax2.bar(model_names, times, color=colors, edgecolor="#21312f", linewidth=0.8, alpha=0.9)
    ax2.set_title("Training Duration (Lower is Faster)", fontsize=10.5, weight="bold", color="#21312f")
    ax2.set_ylabel("Training Time (Seconds)", fontsize=9, weight="bold", color="#21312f")
    for i, v in enumerate(times):
        ax2.text(i, v + (max(times)*0.02), f"{v}s", ha="center", fontsize=8.5, weight="bold")
    ax2.tick_params(axis="x", rotation=15)
    
    plt.tight_layout()
    plot1_path = config.PLOTS_DIR / "modern_boosting_comparison.png"
    fig.savefig(plot1_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    
    # -------------------------------------------------------------
    # DIAGRAM 2: XGBoost Feature Importance
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=120)
    best_model = models.get("xgboost") or models.get("lightgbm") or models.get("hist_gb")
    
    if hasattr(best_model, "feature_importances_"):
        imps = pd.Series(best_model.feature_importances_, index=features).sort_values(ascending=True)
        ax.barh(imps.index, imps.values * 100, color="#0f9f8f", edgecolor="#21312f", linewidth=0.8)
        ax.set_title("XGBoost / Modern Tree Feature Importances (Gain / Weight)", fontsize=11, weight="bold", color="#21312f", pad=12)
        ax.set_xlabel("Relative Importance (%)", fontsize=9.5, weight="bold", color="#21312f")
        ax.set_ylabel("Predictor Feature", fontsize=9.5, weight="bold", color="#21312f")
    else:
        ax.text(0.5, 0.5, "Feature importance unavailable", ha="center")
        
    plt.tight_layout()
    plot2_path = config.PLOTS_DIR / "xgboost_feature_importance.png"
    fig.savefig(plot2_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    
    return {
        "plot_comparison": "modern_boosting_comparison.png",
        "plot_feature_importance": "xgboost_feature_importance.png"
    }


def predict_placement_modern(model_data, inputs, chosen_model="xgboost"):
    """Predict placement using XGBoost, LightGBM, CatBoost or HistGB."""
    models = model_data["models"]
    model = models.get(chosen_model) or next(iter(models.values()))
    features = model_data["features"]
    
    row = pd.DataFrame([{f: float(inputs.get(f, 0.0)) for f in features}])
    prob_placed = float(model.predict_proba(row)[0, 1]) * 100
    pred_class = int(model.predict(row)[0])
    
    return {
        "placement_status": "Placed" if pred_class == 1 else "Not Placed",
        "prob_placed": round(prob_placed, 1),
        "prob_not_placed": round(100.0 - prob_placed, 1),
        "model_name": chosen_model.upper(),
        "inputs": inputs
    }


def save_modern_boosting_report(modern_data):
    """Save Modern Boosted Trees benchmark report to Output/Report/modern_boosting_report.txt."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rep_path = config.REPORTS_DIR / "modern_boosting_report.txt"
    
    with open(rep_path, "w") as f:
        f.write("MODERN BOOSTED TREES BENCHMARK REPORT (50,000 PLACEMENT RECORDS)\n")
        f.write("=" * 75 + "\n")
        f.write(f"{'Framework':<22} {'Accuracy':<12} {'F1':<10} {'ROC-AUC':<12} {'Train Time':<12} {'Latency'}\n")
        f.write("-" * 75 + "\n")
        for b in modern_data["benchmarks"]:
            f.write(f"{b['model']:<22} {b['accuracy']:>6.2f}%     {b['f1']:>6.2f}%    {b['roc_auc']:>6.2f}%     {b['train_time_sec']:>6.3f}s     {b['infer_latency_ms']:>6.3f}ms\n")
        f.write("\nGenerated Plots:\n")
        f.write("  - modern_boosting_comparison.png\n")
        f.write("  - xgboost_feature_importance.png\n")
        
    return str(rep_path)


if __name__ == "__main__":
    print("=" * 70)
    print("  MODERN BOOSTED TREES - PLACEMENT PREDICTION BENCHMARK")
    print("=" * 70)
    df = load_cleaned()
    modern_data = train_modern_boosting_benchmarks(df=df)
    for b in modern_data["benchmarks"]:
        print(f"{b['model']:<22} Acc: {b['accuracy']}% | F1: {b['f1']}% | Time: {b['train_time_sec']}s")
    diagrams = generate_modern_boosting_diagrams(model_data=modern_data)
    print(f"Generated diagrams in: {config.PLOTS_DIR}")
    rep = save_modern_boosting_report(modern_data)
    print(f"Report saved: {rep}")
    print("=" * 70)
