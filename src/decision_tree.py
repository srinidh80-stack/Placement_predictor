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
