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
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import config
from src.data_utils import load_cleaned

BOOSTING_FEATURES = [
    "CGPA", "CodingTestScore", "MockInterviewScore", "AptitudeTestScore",
    "AttendancePercent", "SoftSkillsRating", "Internships", "Projects"
]
TARGET_CLASSIFICATION = "PlacementStatus"


def get_boosting_data(df=None):
    """Load and prepare data for Boosting models."""
    data = load_cleaned() if df is None else df.copy()
    available_features = [f for f in BOOSTING_FEATURES if f in data.columns]
    for col in available_features:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data[col] = data[col].fillna(data[col].median() if not pd.isna(data[col].median()) else 0)
    return data, available_features


def train_boosting_models(df=None, n_estimators=100, learning_rate=0.1, test_size=0.2, random_state=42):
    """Train AdaBoost and Gradient Boosting Classifiers and compute evaluation metrics."""
    data, features = get_boosting_data(df)
    X = data[features]
    y = data[TARGET_CLASSIFICATION].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # AdaBoost with Decision Stump base estimator
    adaboost = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state
    )
    adaboost.fit(X_train, y_train)
    
    # Gradient Boosting (Gradient descent in function space)
    gb = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=3,
        random_state=random_state
    )
    gb.fit(X_train, y_train)
    
    y_pred_ada = adaboost.predict(X_test)
    y_prob_ada = adaboost.predict_proba(X_test)[:, 1]
    
    y_pred_gb = gb.predict(X_test)
    y_prob_gb = gb.predict_proba(X_test)[:, 1]
    
    metrics = {
        "adaboost": {
            "accuracy": round(float(accuracy_score(y_test, y_pred_ada) * 100), 2),
            "precision": round(float(precision_score(y_test, y_pred_ada, zero_division=0) * 100), 2),
            "recall": round(float(recall_score(y_test, y_pred_ada, zero_division=0) * 100), 2),
            "f1": round(float(f1_score(y_test, y_pred_ada, zero_division=0) * 100), 2),
            "roc_auc": round(float(roc_auc_score(y_test, y_prob_ada) * 100), 2),
            "confusion_matrix": confusion_matrix(y_test, y_pred_ada).tolist(),
        },
        "gradient_boosting": {
            "accuracy": round(float(accuracy_score(y_test, y_pred_gb) * 100), 2),
            "precision": round(float(precision_score(y_test, y_pred_gb, zero_division=0) * 100), 2),
            "recall": round(float(recall_score(y_test, y_pred_gb, zero_division=0) * 100), 2),
            "f1": round(float(f1_score(y_test, y_pred_gb, zero_division=0) * 100), 2),
            "roc_auc": round(float(roc_auc_score(y_test, y_prob_gb) * 100), 2),
            "confusion_matrix": confusion_matrix(y_test, y_pred_gb).tolist(),
        },
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
    }
    
    return {
        "adaboost": adaboost,
        "gradient_boosting": gb,
        "metrics": metrics,
        "features": features,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def generate_boosting_diagrams(model_data=None, df=None):
    """Generate staged accuracy and learning rate impact curves."""
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if model_data is None:
        model_data = train_boosting_models(df=df)
        
    adaboost = model_data["adaboost"]
    gb = model_data["gradient_boosting"]
    X_train = model_data["X_train"]
    y_train = model_data["y_train"]
    X_test = model_data["X_test"]
    y_test = model_data["y_test"]
    
    sns.set_theme(style="whitegrid")
    
    # -------------------------------------------------------------
    # DIAGRAM 1: Staged Validation Accuracy Curve Across Boosting Rounds
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=120)
    
    ada_staged = [accuracy_score(y_test, pred) * 100 for pred in adaboost.staged_predict(X_test)]
    gb_staged = [accuracy_score(y_test, pred) * 100 for pred in gb.staged_predict(X_test)]
    rounds = list(range(1, len(ada_staged) + 1))
    
    ax.plot(rounds, ada_staged, color="#3a6b88", linewidth=2.2, label=f"AdaBoost (Final: {ada_staged[-1]:.2f}%)")
    ax.plot(rounds, gb_staged, color="#0f9f8f", linewidth=2.2, label=f"Gradient Boosting (Final: {gb_staged[-1]:.2f}%)")
    
    ax.set_title("Boosting Progression: Test Accuracy over Sequential Rounds", fontsize=11, weight="bold", color="#21312f", pad=12)
    ax.set_xlabel("Number of Boosting Stages / Estimators", fontsize=9.5, weight="bold", color="#21312f")
    ax.set_ylabel("Validation Accuracy (%)", fontsize=9.5, weight="bold", color="#21312f")
    ax.legend(frameon=True, loc="lower right", fontsize=8.5)
    plt.tight_layout()
    plot1_path = config.PLOTS_DIR / "boosting_staged_accuracy.png"
    fig.savefig(plot1_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    
    # -------------------------------------------------------------
    # DIAGRAM 2: Learning Rate vs Convergence Curve
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=120)
    
    # Fast evaluation subset
    sample_size = min(3500, len(X_train))
    sub_idx = np.random.RandomState(42).choice(len(X_train), size=sample_size, replace=False)
    X_sub, y_sub = X_train.iloc[sub_idx], y_train.iloc[sub_idx]
    
    lrs = [0.01, 0.05, 0.1, 0.3, 0.7]
    lr_colors = ["#2f6288", "#0f9f8f", "#f5b84b", "#e05242", "#8b4f88"]
    
    for lr, col in zip(lrs, lr_colors):
        temp_gb = GradientBoostingClassifier(n_estimators=60, learning_rate=lr, max_depth=2, random_state=42)
        temp_gb.fit(X_sub, y_sub)
        scores = [accuracy_score(y_test, p) * 100 for p in temp_gb.staged_predict(X_test)]
        ax.plot(range(1, len(scores) + 1), scores, label=f"lr = {lr}", color=col, linewidth=1.8)
        
    ax.set_title("Gradient Boosting: Learning Rate (Shrinkage) vs Accuracy", fontsize=11, weight="bold", color="#21312f", pad=12)
    ax.set_xlabel("Boosting Stage", fontsize=9.5, weight="bold", color="#21312f")
    ax.set_ylabel("Test Accuracy (%)", fontsize=9.5, weight="bold", color="#21312f")
    ax.legend(frameon=True, loc="lower right", fontsize=8.5)
    plt.tight_layout()
    plot2_path = config.PLOTS_DIR / "boosting_learning_rate.png"
    fig.savefig(plot2_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    
    return {
        "plot_staged_accuracy": "boosting_staged_accuracy.png",
        "plot_learning_rate": "boosting_learning_rate.png"
    }


def predict_placement_boosting(model_data, inputs, model_type="gradient_boosting"):
    """Predict placement using AdaBoost or Gradient Boosting."""
    model = model_data[model_type]
    features = model_data["features"]
    
    row = pd.DataFrame([{f: float(inputs.get(f, 0.0)) for f in features}])
    prob_placed = float(model.predict_proba(row)[0, 1]) * 100
    pred_class = int(model.predict(row)[0])
    
    display_name = "Gradient Boosting" if model_type == "gradient_boosting" else "AdaBoost"
    return {
        "placement_status": "Placed" if pred_class == 1 else "Not Placed",
        "prob_placed": round(prob_placed, 1),
        "prob_not_placed": round(100.0 - prob_placed, 1),
        "model_name": display_name,
        "inputs": inputs
    }


def save_boosting_report(boosting_data):
    """Save Boosting comparison report to Output/Report/boosting_report.txt."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    m_ada = boosting_data["metrics"]["adaboost"]
    m_gb = boosting_data["metrics"]["gradient_boosting"]
    rep_path = config.REPORTS_DIR / "boosting_report.txt"
    
    with open(rep_path, "w") as f:
        f.write("BOOSTING FOUNDATIONS REPORT (50,000 PLACEMENT RECORDS)\n")
        f.write("=" * 60 + "\n")
        f.write("AdaBoost Classifier (100 Decision Stumps):\n")
        f.write(f"  Accuracy:  {m_ada['accuracy']}%\n")
        f.write(f"  ROC-AUC:   {m_ada['roc_auc']}%\n")
        f.write(f"  Precision: {m_ada['precision']}%\n")
        f.write(f"  Recall:    {m_ada['recall']}%\n")
        f.write(f"  F1-Score:  {m_ada['f1']}%\n\n")
        f.write("Gradient Boosting Classifier (100 Stages, max_depth=3, lr=0.1):\n")
        f.write(f"  Accuracy:  {m_gb['accuracy']}%\n")
        f.write(f"  ROC-AUC:   {m_gb['roc_auc']}%\n")
        f.write(f"  Precision: {m_gb['precision']}%\n")
        f.write(f"  Recall:    {m_gb['recall']}%\n")
        f.write(f"  F1-Score:  {m_gb['f1']}%\n\n")
        f.write("Generated Plots:\n")
        f.write("  - boosting_staged_accuracy.png\n")
        f.write("  - boosting_learning_rate.png\n")
        
    return str(rep_path)


if __name__ == "__main__":
    print("=" * 70)
    print("  BOOSTING MODELS - PLACEMENT PREDICTION")
    print("=" * 70)
    df = load_cleaned()
    boosting_data = train_boosting_models(df=df, n_estimators=100)
    print(f"AdaBoost Accuracy: {boosting_data['metrics']['adaboost']['accuracy']}%")
    print(f"Gradient Boosting Accuracy: {boosting_data['metrics']['gradient_boosting']['accuracy']}%")
    diagrams = generate_boosting_diagrams(model_data=boosting_data)
    print(f"Generated diagrams in: {config.PLOTS_DIR}")
    rep = save_boosting_report(boosting_data)
    print(f"Report saved: {rep}")
    print("=" * 70)
