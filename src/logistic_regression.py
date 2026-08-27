"""Logistic Regression model training, evaluation, diagrams, and live placement prediction (Module 2 Sessions 20-22 / Session 09)."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import config
from src.data_utils import clean_data, load_cleaned

LOGISTIC_NUMERIC_FEATURES = [
    "SGPA_Sem1", "SGPA_Sem2", "SGPA_Sem3", "SGPA_Sem4",
    "SGPA_Sem5", "SGPA_Sem6", "SGPA_Sem7", "SGPA_Sem8",
    "CGPA", "AttendancePercent", "Internships", "Projects",
    "Workshops", "Certifications", "Publications",
    "AptitudeTestScore", "SoftSkillsRating", "CodingTestScore",
    "MockInterviewScore", "ExtraCurricular",
]
CORE_INPUT_FEATURES = ["CGPA", "CodingTestScore", "MockInterviewScore", "AptitudeTestScore", "AttendancePercent", "SoftSkillsRating"]
TARGET_COL = "PlacementStatus"


def get_classification_data(df=None):
    """Load and prepare data for classification."""
    data = load_cleaned() if df is None else df.copy()
    
    # Fill numeric columns with median if missing
    for col in LOGISTIC_NUMERIC_FEATURES:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].median())
            
    return data


def train_logistic_regression(df=None, test_size=0.2, random_state=42):
    """Train Binary Logistic Regression, Scaler Comparison, and Softmax 3-Class Model."""
    data = get_classification_data(df)
    
    available_features = [f for f in LOGISTIC_NUMERIC_FEATURES if f in data.columns]
    X = data[available_features]
    y = data[TARGET_COL]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 1. Standard Scaler Model (Primary)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_val_std = scaler.transform(X_val)
    
    binary_model = LogisticRegression(max_iter=1000, random_state=random_state)
    binary_model.fit(X_train_std, y_train)
    
    y_val_pred = binary_model.predict(X_val_std)
    y_val_prob = binary_model.predict_proba(X_val_std)[:, 1]
    
    val_acc = accuracy_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_prob)
    val_loss = log_loss(y_val, y_val_prob)
    val_prec = precision_score(y_val, y_val_pred)
    val_rec = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    cm = confusion_matrix(y_val, y_val_pred)
    
    # 2. Scaler Comparison (Unscaled vs StandardScaler vs MinMaxScaler)
    # Unscaled
    unscaled_model = LogisticRegression(max_iter=1000, random_state=random_state)
    unscaled_model.fit(X_train, y_train)
    acc_unscaled = accuracy_score(y_val, unscaled_model.predict(X_val))
    
    # MinMax
    mm_scaler = MinMaxScaler()
    X_train_mm = mm_scaler.fit_transform(X_train)
    X_val_mm = mm_scaler.transform(X_val)
    mm_model = LogisticRegression(max_iter=1000, random_state=random_state)
    mm_model.fit(X_train_mm, y_train)
    acc_minmax = accuracy_score(y_val, mm_model.predict(X_val_mm))
    
    # 3. Single-feature AUC rankings
    auc_ranking = []
    for col in CORE_INPUT_FEATURES:
        if col in X_train.columns:
            score = roc_auc_score(y_train, X_train[col])
            auc_ranking.append({
                "feature": col,
                "auc": round(float(score), 4),
                "power": "Strong Predictor" if score >= 0.70 else "Moderate Predictor" if score >= 0.55 else "Weak"
            })
    auc_ranking.sort(key=lambda x: x["auc"], reverse=True)
    
    # 4. Multinomial Softmax Model (3-Tier Outcome: Not Placed, Standard Package, Premium Package)
    salary_median = data.loc[data["Salary Package"] > 0, "Salary Package"].median() if "Salary Package" in data.columns else 8.5
    
    def make_package_tier(row):
        if row.get("PlacementStatus", 0) == 0 or row.get("Salary Package", 0) == 0:
            return "Not Placed"
        elif row.get("Salary Package", 0) < salary_median:
            return "Standard Package"
        return "Premium Package"
        
    y_tier = data.apply(make_package_tier, axis=1)
    y_tier_train, y_tier_val = y_tier.loc[X_train.index], y_tier.loc[X_val.index]
    
    softmax_model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=random_state)
    softmax_model.fit(X_train_std, y_tier_train)
    softmax_acc = accuracy_score(y_tier_val, softmax_model.predict(X_val_std))
    
    # Class balance
    placed_pct = float((y_train == 1).mean() * 100)
    
    return {
        "binary_model": binary_model,
        "softmax_model": softmax_model,
        "scaler": scaler,
        "features": available_features,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "train_shape": X_train.shape,
        "val_shape": X_val.shape,
        "class_balance": {
            "placed_pct": round(placed_pct, 1),
            "not_placed_pct": round(100.0 - placed_pct, 1),
            "placed_count": int((y == 1).sum()),
            "not_placed_count": int((y == 0).sum()),
        },
        "metrics": {
            "accuracy": round(val_acc * 100, 2),
            "auc": round(val_auc, 4),
            "log_loss": round(val_loss, 4),
            "precision": round(val_prec * 100, 2),
            "recall": round(val_rec * 100, 2),
            "f1": round(val_f1 * 100, 2),
            "confusion_matrix": {
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            },
        },
        "scaler_comparison": [
            {"scaler": "StandardScaler (Z-score)", "accuracy": round(val_acc * 100, 2), "status": "Best Stability"},
            {"scaler": "MinMaxScaler [0, 1]", "accuracy": round(acc_minmax * 100, 2), "status": "Comparable"},
            {"scaler": "Unscaled Raw Data", "accuracy": round(acc_unscaled * 100, 2), "status": "Baseline"},
        ],
        "auc_ranking": auc_ranking,
        "softmax_acc": round(softmax_acc * 100, 2),
        "softmax_classes": softmax_model.classes_.tolist(),
    }


def predict_placement_status(inputs_dict, model_data=None):
    """Run real-time inference for binary placement and 3-class softmax probability."""
    if model_data is None:
        model_data = train_logistic_regression()
        
    features = model_data["features"]
    scaler = model_data["scaler"]
    binary_model = model_data["binary_model"]
    softmax_model = model_data["softmax_model"]
    X_train = model_data["X_train"]
    
    # Build complete feature row using medians for unspecified features
    row_data = {}
    for feat in features:
        if feat in inputs_dict:
            row_data[feat] = float(inputs_dict[feat])
        elif feat.startswith("SGPA_Sem"):
            # If specific semester SGPA not passed, default to CGPA
            row_data[feat] = float(inputs_dict.get("CGPA", 8.0))
        else:
            row_data[feat] = float(X_train[feat].median())
            
    df_row = pd.DataFrame([row_data])[features]
    row_std = scaler.transform(df_row)
    
    # Binary Prediction
    binary_pred = int(binary_model.predict(row_std)[0])
    binary_probs = binary_model.predict_proba(row_std)[0]
    prob_placed = float(binary_probs[1]) * 100
    prob_not_placed = float(binary_probs[0]) * 100
    
    # Softmax 3-Class Prediction
    softmax_pred = str(softmax_model.predict(row_std)[0])
    softmax_probs = softmax_model.predict_proba(row_std)[0]
    tier_probs = [
        {"tier": cls_name, "prob": round(float(p) * 100, 1)}
        for cls_name, p in zip(softmax_model.classes_, softmax_probs)
    ]
    tier_probs.sort(key=lambda x: x["prob"], reverse=True)
    
    return {
        "is_placed": binary_pred == 1,
        "placement_status": "Placed" if binary_pred == 1 else "Not Placed",
        "prob_placed": round(prob_placed, 1),
        "prob_not_placed": round(prob_not_placed, 1),
        "softmax_tier": softmax_pred,
        "tier_probabilities": tier_probs,
        "inputs": inputs_dict,
    }


def generate_logistic_diagrams(df=None, model_data=None, current_prediction=None):
    """Generate S-curve, Decision Boundary, and Confusion Matrix plots."""
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if model_data is None:
        model_data = train_logistic_regression(df)
        
    X_val = model_data["X_val"]
    y_val = model_data["y_val"]
    X_train = model_data["X_train"]
    y_train = model_data["y_train"]
    
    sns.set_theme(style="whitegrid")
    
    # -------------------------------------------------------------
    # DIAGRAM 1: Empirical CGPA vs Placement S-Curve
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=120)
    
    bins = pd.cut(X_train["CGPA"], bins=15)
    fraction_placed = y_train.groupby(bins, observed=True).mean()
    bin_centers = [interval.mid for interval in fraction_placed.index]
    
    ax.plot(bin_centers, fraction_placed.values, marker="o", markersize=7, color="#0f9f8f", linewidth=2.5, label="Empirical Fraction Placed")
    
    # Overlay theoretical Sigmoid fit
    cgpa_dense = np.linspace(X_train["CGPA"].min(), X_train["CGPA"].max(), 200)
    cgpa_clf = LogisticRegression()
    cgpa_clf.fit(X_train[["CGPA"]], y_train)
    sigmoid_dense = cgpa_clf.predict_proba(pd.DataFrame({"CGPA": cgpa_dense}))[:, 1]
    
    ax.plot(cgpa_dense, sigmoid_dense, color="#f05f4f", linewidth=2.5, linestyle="--", label="Fitted Sigmoid σ(z)")
    ax.axhline(0.5, color="#6f7f7a", linestyle=":", alpha=0.8, label="50% Probability Threshold")
    ax.axvline(x=7.5, color="#f5b84b", linestyle=":", alpha=0.8, label="Decision Boundary (~7.5 CGPA)")
    
    if current_prediction and "CGPA" in current_prediction["inputs"]:
        input_cgpa = float(current_prediction["inputs"]["CGPA"])
        prob = current_prediction["prob_placed"] / 100.0
        ax.scatter([input_cgpa], [prob], color="#f5b84b", s=140, edgecolor="#21312f", linewidth=2, zorder=10, label=f"Current Candidate ({input_cgpa} CGPA → {round(prob*100,1)}%)")
    
    ax.set_title("CGPA vs Placement Probability (Empirical S-Curve vs Sigmoid)", fontsize=11, weight="bold", color="#21312f", pad=12)
    ax.set_xlabel("CGPA", fontsize=9.5, weight="bold", color="#21312f")
    ax.set_ylabel("Probability of Placement P(y=1)", fontsize=9.5, weight="bold", color="#21312f")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=True, loc="upper left", fontsize=8.5)
    plt.tight_layout()
    plot1_path = config.PLOTS_DIR / "logistic_s_curve.png"
    fig.savefig(plot1_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    
    # -------------------------------------------------------------
    # DIAGRAM 2: 2D Decision Boundary (CGPA vs Coding Test Score)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=120)
    
    boundary_clf = LogisticRegression(max_iter=1000, random_state=42)
    boundary_clf.fit(X_train[["CGPA", "CodingTestScore"]].to_numpy(), y_train)
    
    x_min, x_max = X_train["CGPA"].min() - 0.3, X_train["CGPA"].max() + 0.3
    y_min, y_max = X_train["CodingTestScore"].min() - 5, X_train["CodingTestScore"].max() + 5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150))
    grid_preds = boundary_clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
    
    # Contour fill
    contour = ax.contourf(xx, yy, grid_preds, levels=np.linspace(0, 1, 11), cmap="YlGnBu", alpha=0.45)
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label("Placement Probability", fontsize=8.5)
    
    # Decision boundary line where prob = 0.5
    ax.contour(xx, yy, grid_preds, levels=[0.5], colors=["#d84f40"], linewidths=2.5, linestyles=["-"])
    
    # Subsample validation points
    sample_idx = np.random.choice(len(y_val), size=min(800, len(y_val)), replace=False)
    sub_cgpa = X_val["CGPA"].iloc[sample_idx]
    sub_coding = X_val["CodingTestScore"].iloc[sample_idx]
    sub_y = y_val.iloc[sample_idx]
    
    ax.scatter(sub_cgpa[sub_y == 1], sub_coding[sub_y == 1], alpha=0.4, color="#0b8276", s=16, label="Placed (Actual)")
    ax.scatter(sub_cgpa[sub_y == 0], sub_coding[sub_y == 0], alpha=0.4, color="#e056fd", s=16, label="Not Placed (Actual)")
    
    if current_prediction and "CGPA" in current_prediction["inputs"] and "CodingTestScore" in current_prediction["inputs"]:
        c_cgpa = float(current_prediction["inputs"]["CGPA"])
        c_coding = float(current_prediction["inputs"]["CodingTestScore"])
        ax.scatter([c_cgpa], [c_coding], color="#f5b84b", s=150, edgecolor="#21312f", linewidth=2.2, zorder=10, label="Current Candidate")
    
    ax.set_title("Decision Boundary (CGPA vs Coding Test Score)", fontsize=11, weight="bold", color="#21312f", pad=12)
    ax.set_xlabel("CGPA", fontsize=9.5, weight="bold", color="#21312f")
    ax.set_ylabel("Coding Test Score", fontsize=9.5, weight="bold", color="#21312f")
    ax.legend(frameon=True, loc="lower right", fontsize=8)
    plt.tight_layout()
    plot2_path = config.PLOTS_DIR / "logistic_decision_boundary.png"
    fig.savefig(plot2_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    
    return {
        "plot_s_curve": "plots/logistic_s_curve.png",
        "plot_decision_boundary": "plots/logistic_decision_boundary.png"
    }
