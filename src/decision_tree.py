from io import BytesIO
import base64
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import config
from src.data_utils import load_cleaned

DT_FEATURES = [
    "CGPA", "CodingTestScore", "MockInterviewScore", "AptitudeTestScore",
    "AttendancePercent", "SoftSkillsRating", "Internships", "Projects"
]
TARGET_CLASSIFICATION = "PlacementStatus"
TARGET_REGRESSION = "Salary Package"


def get_decision_tree_data(df=None):
    """Load and prepare project dataset for Decision Tree modeling."""
    data = load_cleaned() if df is None else df.copy()
    
    available_features = [f for f in DT_FEATURES if f in data.columns]
    for col in available_features:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data[col] = data[col].fillna(data[col].median() if not pd.isna(data[col].median()) else 0)
        
    return data, available_features


def calculate_entropy(labels):
    """Entropy: H(S) = - sum(p_i * log2(p_i))"""
    if len(labels) == 0:
        return 0.0
    counts = pd.Series(labels).value_counts()
    total = len(labels)
    h = 0.0
    for count in counts:
        p = count / total
        if p > 0:
            h -= p * math.log2(p)
    return round(h, 4)


def calculate_gini(labels):
    """Gini Index: Gini(S) = 1 - sum(p_i^2)"""
    if len(labels) == 0:
        return 0.0
    counts = pd.Series(labels).value_counts()
    total = len(labels)
    sum_sq = sum((c / total) ** 2 for c in counts)
    return round(1.0 - sum_sq, 4)


def calculate_mse(values):
    """MSE: (1/n) * sum((y_i - y_bar)^2)"""
    if len(values) == 0:
        return 0.0
    arr = np.array(values, dtype=float)
    return round(float(np.mean((arr - np.mean(arr)) ** 2)), 4)


def train_decision_tree_classifier(df=None, criterion="entropy", max_depth=3, min_samples_split=5):
    """Train Decision Tree Classifier on project PlacementStatus."""
    data, features = get_decision_tree_data(df)
    
    X = data[features]
    y = data[TARGET_CLASSIFICATION].astype(int)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train primary model
    clf = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)
    y_val_prob = clf.predict_proba(X_val)[:, 1] if hasattr(clf, "predict_proba") else y_val_pred
    
    tr_acc = accuracy_score(y_train, y_train_pred)
    va_acc = accuracy_score(y_val, y_val_pred)
    val_prec = precision_score(y_val, y_val_pred, zero_division=0)
    val_rec = recall_score(y_val, y_val_pred, zero_division=0)
    val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
    val_auc = roc_auc_score(y_val, y_val_prob)
    cm = confusion_matrix(y_val, y_val_pred)
    
    # Feature Importances
    importances = [
        {"feature": f, "importance": round(float(imp), 4), "pct": round(float(imp) * 100, 1)}
        for f, imp in zip(features, clf.feature_importances_)
    ]
    importances.sort(key=lambda x: x["importance"], reverse=True)
    
    # Depth Overfitting Curve (Depths 1 to 8)
    depth_curve = []
    for d in range(1, 9):
        tmp_clf = DecisionTreeClassifier(criterion=criterion, max_depth=d, random_state=42)
        tmp_clf.fit(X_train, y_train)
        d_tr = accuracy_score(y_train, tmp_clf.predict(X_train))
        d_va = accuracy_score(y_val, tmp_clf.predict(X_val))
        depth_curve.append({
            "depth": d,
            "train_acc": round(d_tr * 100, 2),
            "val_acc": round(d_va * 100, 2),
            "gap": round((d_tr - d_va) * 100, 2)
        })
        
    root_ent = calculate_entropy(y_train)
    root_gi = calculate_gini(y_train)
    
    return {
        "model": clf,
        "features": features,
        "criterion": criterion,
        "max_depth": max_depth,
        "actual_depth": clf.get_depth(),
        "n_leaves": clf.get_n_leaves(),
        "root_entropy": root_ent,
        "root_gini": root_gi,
        "metrics": {
            "train_accuracy": round(tr_acc * 100, 2),
            "val_accuracy": round(va_acc * 100, 2),
            "precision": round(val_prec * 100, 2),
            "recall": round(val_rec * 100, 2),
            "f1_score": round(val_f1 * 100, 2),
            "auc": round(val_auc, 4),
            "gap": round((tr_acc - va_acc) * 100, 2),
            "confusion_matrix": {
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            }
        },
        "feature_importances": importances,
        "depth_curve": depth_curve,
        "tree_text": export_text(clf, feature_names=features),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
    }


def train_decision_tree_regressor(df=None, max_depth=3):
    """Train Decision Tree Regressor on Salary Package using MSE criterion."""
    data, features = get_decision_tree_data(df)
    
    if TARGET_REGRESSION not in data.columns:
        return None
        
    placed_data = data[data[TARGET_REGRESSION] > 0].copy()
    if len(placed_data) < 50:
        return None
        
    X = placed_data[features]
    y = placed_data[TARGET_REGRESSION]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    reg = DecisionTreeRegressor(criterion="squared_error", max_depth=max_depth, random_state=42)
    reg.fit(X_train, y_train)
    
    y_pred = reg.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    
    return {
        "model": reg,
        "max_depth": max_depth,
        "r2_score": round(float(r2), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(rmse), 4),
        "mean_salary": round(float(y.mean()), 2),
        "root_mse": calculate_mse(y_train),
    }


def plot_to_base64(fig):
    """Helper to convert matplotlib figure to base64 image string."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    b64_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{b64_str}"


def generate_decision_tree_diagrams(clf_bundle, reg_bundle=None):
    """Generate visualized Decision Tree plots for dashboard display."""
    diagrams = {}
    
    # 1. Tree Architecture Diagram
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    plot_tree(
        clf_bundle["model"],
        feature_names=clf_bundle["features"],
        class_names=["Not Placed", "Placed"],
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
        precision=2
    )
    ax.set_title(
        f"Placement Decision Tree ({clf_bundle['criterion'].capitalize()} Criterion | max_depth={clf_bundle['max_depth']})",
        fontsize=11, fontweight="bold", color="#0f766e", pad=12
    )
    fig.tight_layout()
    tree_plot_path = config.PLOTS_DIR / "decision_tree_structure.png"
    fig.savefig(tree_plot_path, bbox_inches="tight", dpi=120)
    diagrams["plot_tree"] = "decision_tree_structure.png"
    diagrams["tree_plot"] = plot_to_base64(fig)
    
    # 2. Feature Importance Bar Chart
    importances = clf_bundle["feature_importances"]
    feat_names = [item["feature"] for item in reversed(importances)]
    feat_scores = [item["importance"] for item in reversed(importances)]
    
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    bars = ax.barh(feat_names, feat_scores, color="#0f766e", edgecolor="#042f2e", height=0.6)
    for bar in bars:
        w = bar.get_width()
        if w > 0.005:
            ax.annotate(f"{w*100:.1f}%",
                        xy=(w, bar.get_y() + bar.get_height() / 2),
                        xytext=(4, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=8.5, fontweight="bold")
    ax.set_xlabel("Feature Importance Score (Impurity Reduction)", fontsize=9.5, fontweight="bold")
    ax.set_title("Decision Tree Feature Importance", fontsize=11, fontweight="bold", color="#1e293b")
    ax.set_xlim(0, max(feat_scores) * 1.2 if feat_scores and max(feat_scores) > 0 else 1.0)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    fig.tight_layout()
    dt_imp_path = config.PLOTS_DIR / "dt_feature_importance.png"
    fig.savefig(dt_imp_path, bbox_inches="tight", dpi=120)
    diagrams["plot_feature_importance"] = "dt_feature_importance.png"
    diagrams["feature_importance_plot"] = plot_to_base64(fig)
    
    # 3. Overfitting vs Depth Curve
    depth_curve = clf_bundle["depth_curve"]
    depths = [d["depth"] for d in depth_curve]
    tr_accs = [d["train_acc"] for d in depth_curve]
    va_accs = [d["val_acc"] for d in depth_curve]
    
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(depths, tr_accs, marker="o", color="#2563eb", linewidth=2, label="Train Accuracy (%)")
    ax.plot(depths, va_accs, marker="s", color="#16a34a", linewidth=2, label="Validation Accuracy (%)")
    ax.fill_between(depths, tr_accs, va_accs, color="#ef4444", alpha=0.1, label="Overfitting Gap")
    
    ax.set_xlabel("Tree Depth (max_depth)", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=9.5, fontweight="bold")
    ax.set_title("Decision Tree Overfitting Diagnostic (Train vs Val Accuracy)", fontsize=11, fontweight="bold", color="#1e293b")
    ax.set_xticks(depths)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    dt_depth_path = config.PLOTS_DIR / "dt_depth_curve.png"
    fig.savefig(dt_depth_path, bbox_inches="tight", dpi=120)
    diagrams["plot_depth_curve"] = "dt_depth_curve.png"
    diagrams["overfitting_plot"] = plot_to_base64(fig)
    
    return diagrams


def predict_placement_tree(inputs, clf_bundle):
    """Predict placement status for a single student using the trained Decision Tree."""
    model = clf_bundle["model"]
    features = clf_bundle["features"]
    
    row = []
    for f in features:
        val = inputs.get(f, 0.0)
        try:
            row.append(float(val))
        except (ValueError, TypeError):
            row.append(0.0)
            
    X_input = pd.DataFrame([row], columns=features)
    pred = int(model.predict(X_input)[0])
    prob = float(model.predict_proba(X_input)[0][1]) if hasattr(model, "predict_proba") else (1.0 if pred == 1 else 0.0)
    
    return {
        "prediction": "Placed" if pred == 1 else "Not Placed",
        "is_placed": pred == 1,
        "probability": round(prob * 100, 1),
        "status_color": "#16a34a" if pred == 1 else "#dc2626",
    }


def analyze_pruning_and_splitting(df=None, random_state=42):
    """Analyze pre-pruning and cost-complexity pruning on 50k placement records."""
    data, features = get_decision_tree_data(df)
    X = data[features]
    y = data[TARGET_CLASSIFICATION].astype(int)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    # 1. Full unconstrained tree (overfitting baseline)
    full_tree = DecisionTreeClassifier(criterion="entropy", random_state=random_state)
    full_tree.fit(X_train, y_train)
    full_tr_acc = accuracy_score(y_train, full_tree.predict(X_train)) * 100
    full_val_acc = accuracy_score(y_val, full_tree.predict(X_val)) * 100
    full_depth = full_tree.get_depth()
    full_leaves = full_tree.get_n_leaves()
    
    # 2. Pre-pruned tree (depth=3, min_samples_split=20, min_samples_leaf=10)
    pruned_tree = DecisionTreeClassifier(
        criterion="entropy", max_depth=3, min_samples_split=20, min_samples_leaf=10, random_state=random_state
    )
    pruned_tree.fit(X_train, y_train)
    pruned_tr_acc = accuracy_score(y_train, pruned_tree.predict(X_train)) * 100
    pruned_val_acc = accuracy_score(y_val, pruned_tree.predict(X_val)) * 100
    pruned_depth = pruned_tree.get_depth()
    pruned_leaves = pruned_tree.get_n_leaves()
    
    # 3. Cost-Complexity Pruning (ccp_alpha path on subsample for efficient evaluation)
    sample_size = min(5000, len(X_train))
    idx = np.random.RandomState(random_state).choice(len(X_train), size=sample_size, replace=False)
    path = full_tree.cost_complexity_pruning_path(X_train.iloc[idx], y_train.iloc[idx])
    ccp_alphas = path.ccp_alphas[::max(1, len(path.ccp_alphas) // 15)]  # sample 15 alphas
    
    alpha_val_accs = []
    alpha_train_accs = []
    for a in ccp_alphas:
        t = DecisionTreeClassifier(random_state=random_state, ccp_alpha=a)
        t.fit(X_train.iloc[idx], y_train.iloc[idx])
        alpha_train_accs.append(accuracy_score(y_train, t.predict(X_train)) * 100)
        alpha_val_accs.append(accuracy_score(y_val, t.predict(X_val)) * 100)
        
    # Generate Pruning Curve Plot
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=120)
    ax.plot(ccp_alphas, alpha_train_accs, marker="o", color="#2563eb", linewidth=2, label="Train Accuracy")
    ax.plot(ccp_alphas, alpha_val_accs, marker="s", color="#16a34a", linewidth=2, label="Validation Accuracy")
    ax.set_title("Cost-Complexity Pruning: Effective Alpha vs Accuracy (50k Dataset)", fontsize=11, weight="bold", color="#1e293b", pad=12)
    ax.set_xlabel("Effective Alpha (ccp_alpha)", fontsize=9.5, weight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=9.5, weight="bold")
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    pruning_plot_path = config.PLOTS_DIR / "pruning_curve.png"
    fig.savefig(pruning_plot_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    
    # Write splitting and decision tree reports
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    dt_rep_path = config.REPORTS_DIR / "decision_tree_report.txt"
    with open(dt_rep_path, "w") as f:
        f.write("DECISION TREE & PRUNING REPORT (50,000 PLACEMENT RECORDS)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Unconstrained Tree:\n")
        f.write(f"  Depth: {full_depth}, Leaves: {full_leaves}\n")
        f.write(f"  Train Accuracy: {full_tr_acc:.2f}%, Val Accuracy: {full_val_acc:.2f}%\n")
        f.write(f"  Overfitting Gap: {full_tr_acc - full_val_acc:.2f}%\n\n")
        f.write(f"Pruned Tree (max_depth=3, min_samples_leaf=10):\n")
        f.write(f"  Depth: {pruned_depth}, Leaves: {pruned_leaves}\n")
        f.write(f"  Train Accuracy: {pruned_tr_acc:.2f}%, Val Accuracy: {pruned_val_acc:.2f}%\n")
        f.write(f"  Overfitting Gap: {pruned_tr_acc - pruned_val_acc:.2f}%\n")
        
    crit_rep_path = config.REPORTS_DIR / "splitting_criteria_report.txt"
    with open(crit_rep_path, "w") as f:
        f.write("SPLITTING CRITERIA COMPARISON (ENTROPY vs GINI vs MSE)\n")
        f.write("=" * 60 + "\n")
        f.write("Classification Criteria:\n")
        f.write("  - Entropy: H(S) = - sum(p_i * log2(p_i)). Information Gain = H(parent) - H(split)\n")
        f.write("  - Gini Impurity: Gini(S) = 1 - sum(p_i^2). Computationally faster (no logarithm)\n\n")
        f.write("Regression Criterion:\n")
        f.write("  - Mean Squared Error (MSE): MSE = (1/n) * sum((y_i - y_mean)^2)\n")
        f.write(f"Pruning Plot generated: {pruning_plot_path.name}\n")
        
    return {
        "full_depth": full_depth,
        "full_val_acc": round(full_val_acc, 2),
        "pruned_depth": pruned_depth,
        "pruned_val_acc": round(pruned_val_acc, 2),
        "pruning_plot": "pruning_curve.png",
        "decision_tree_report": str(dt_rep_path),
        "splitting_report": str(crit_rep_path)
    }


if __name__ == "__main__":
    print("=" * 70)
    print("  DECISION TREE - PLACEMENT PREDICTION DATASET")
    print("=" * 70)
    
    print("\n1. Loading project dataset...")
    df = load_cleaned()
    print(f"Dataset shape: {df.shape}")
    
    clf = train_decision_tree_classifier(df=df, criterion="entropy", max_depth=3)
    print(f"\nDecision Tree (Entropy, Depth 3) -> Validation Acc: {clf['metrics']['val_accuracy']}%")
    print(f"Root Impurity (Entropy): {clf['root_entropy']}")
    print(f"Total Leaves: {clf['n_leaves']}")
    
    diagrams = generate_decision_tree_diagrams(clf)
    print(f"Generated diagrams in: {config.PLOTS_DIR}")
    
    pruning_res = analyze_pruning_and_splitting(df)
    print(f"Pruning analysis complete: full depth {pruning_res['full_depth']} vs pruned depth {pruning_res['pruned_depth']}")
    print(f"Reports written to: {config.REPORTS_DIR}")
    print("=" * 70)
