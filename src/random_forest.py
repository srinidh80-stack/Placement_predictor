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
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

import config
from src.data_utils import load_cleaned

RF_FEATURES = [
    "CGPA", "CodingTestScore", "MockInterviewScore", "AptitudeTestScore",
    "AttendancePercent", "SoftSkillsRating", "Internships", "Projects",
    "Workshops", "Certifications"
]
TARGET_CLASSIFICATION = "PlacementStatus"
TARGET_REGRESSION = "Salary Package"


def get_rf_data(df=None):
    """Load and prepare data for Random Forest and Bagging models."""
    data = load_cleaned() if df is None else df.copy()
    available_features = [f for f in RF_FEATURES if f in data.columns]
    for col in available_features:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data[col] = data[col].fillna(data[col].median() if not pd.isna(data[col].median()) else 0)
    return data, available_features


def train_random_forest_classifier(df=None, n_estimators=100, max_depth=None, max_features="sqrt", test_size=0.2, random_state=42):
    """Train Random Forest Classifier with Out-of-Bag (OOB) score and evaluation metrics."""
    data, features = get_rf_data(df)
    
    X = data[features]
    y = data[TARGET_CLASSIFICATION].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        oob_score=True,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # Bagging Classifier comparison (using standard DecisionTreeClassifier as base)
    bagging = BaggingClassifier(
        n_estimators=n_estimators,
        oob_score=True,
        random_state=random_state,
        n_jobs=-1
    )
    bagging.fit(X_train, y_train)
    
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    
    y_pred_bag = bagging.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred_rf).tolist()
    
    importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    
    metrics = {
        "rf_accuracy": round(float(accuracy_score(y_test, y_pred_rf) * 100), 2),
        "rf_precision": round(float(precision_score(y_test, y_pred_rf, zero_division=0) * 100), 2),
        "rf_recall": round(float(recall_score(y_test, y_pred_rf, zero_division=0) * 100), 2),
        "rf_f1": round(float(f1_score(y_test, y_pred_rf, zero_division=0) * 100), 2),
        "rf_roc_auc": round(float(roc_auc_score(y_test, y_prob_rf) * 100), 2),
        "rf_oob_score": round(float(rf.oob_score_ * 100), 2),
        "bagging_accuracy": round(float(accuracy_score(y_test, y_pred_bag) * 100), 2),
        "bagging_oob_score": round(float(bagging.oob_score_ * 100), 2),
        "confusion_matrix": cm,
        "feature_importances": importances.to_dict(),
        "top_features": [{"feature": k, "importance": round(float(v * 100), 2)} for k, v in importances.items()],
        "n_estimators": n_estimators,
        "max_depth": max_depth or "None (Full)",
    }
    
    return {
        "model": rf,
        "bagging_model": bagging,
        "metrics": metrics,
        "features": features,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_importances": importances
    }


def generate_random_forest_diagrams(model_data=None, df=None):
    """Generate Feature Importance and n_estimators tuning curve plots."""
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if model_data is None:
        model_data = train_random_forest_classifier(df=df)
        
    features = model_data["features"]
    importances = model_data["feature_importances"]
    X_train = model_data["X_train"]
    y_train = model_data["y_train"]
    X_test = model_data["X_test"]
    y_test = model_data["y_test"]
    
    sns.set_theme(style="whitegrid")
    
    # -------------------------------------------------------------
    # DIAGRAM 1: Random Forest Feature Importances (Gini Importance)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=120)
    sorted_imp = importances.sort_values(ascending=True)
    colors = ["#0f9f8f" if i >= len(sorted_imp) - 3 else "#3a6b88" for i in range(len(sorted_imp))]
    ax.barh(sorted_imp.index, sorted_imp.values * 100, color=colors, edgecolor="#21312f", linewidth=0.8)
    ax.set_title("Random Forest Feature Importances (Mean Decrease in Impurity)", fontsize=11, weight="bold", color="#21312f", pad=12)
    ax.set_xlabel("Relative Importance (%)", fontsize=9.5, weight="bold", color="#21312f")
    ax.set_ylabel("Predictor Feature", fontsize=9.5, weight="bold", color="#21312f")
    plt.tight_layout()
    plot1_path = config.PLOTS_DIR / "rf_feature_importance.png"
    fig.savefig(plot1_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    
    # -------------------------------------------------------------
    # DIAGRAM 2: n_estimators vs OOB Score and Test Accuracy Curve
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=120)
    
    # Subsample for faster curve rendering
    sample_size = min(4000, len(X_train))
    sample_idx = np.random.RandomState(42).choice(len(X_train), size=sample_size, replace=False)
    X_sub, y_sub = X_train.iloc[sample_idx], y_train.iloc[sample_idx]
    
    tree_counts = [10, 25, 50, 75, 100, 150]
    oob_scores = []
    val_accs = []
    
    for n in tree_counts:
        sub_rf = RandomForestClassifier(n_estimators=n, oob_score=True, random_state=42, n_jobs=-1)
        sub_rf.fit(X_sub, y_sub)
        oob_scores.append(sub_rf.oob_score_ * 100)
        val_accs.append(accuracy_score(y_test, sub_rf.predict(X_test)) * 100)
        
    ax.plot(tree_counts, oob_scores, marker="o", linewidth=2.2, color="#0f9f8f", label="Out-of-Bag (OOB) Accuracy")
    ax.plot(tree_counts, val_accs, marker="s", linewidth=2.2, color="#e05242", linestyle="--", label="Validation Set Accuracy")
    ax.set_title("Effect of Ensemble Size (n_estimators) on Stability and Accuracy", fontsize=11, weight="bold", color="#21312f", pad=12)
    ax.set_xlabel("Number of Trees in Forest (n_estimators)", fontsize=9.5, weight="bold", color="#21312f")
    ax.set_ylabel("Accuracy Score (%)", fontsize=9.5, weight="bold", color="#21312f")
    ax.legend(frameon=True, loc="lower right", fontsize=8.5)
    plt.tight_layout()
    plot2_path = config.PLOTS_DIR / "rf_estimators_curve.png"
    fig.savefig(plot2_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    
    return {
        "plot_feature_importance": "rf_feature_importance.png",
        "plot_estimators_curve": "rf_estimators_curve.png"
    }


def predict_placement_rf(model_data, inputs):
    """Predict placement probability and classification for a student using Random Forest."""
    rf = model_data["model"]
    features = model_data["features"]
    
    row = pd.DataFrame([{f: float(inputs.get(f, 0.0)) for f in features}])
    prob_placed = float(rf.predict_proba(row)[0, 1]) * 100
    pred_class = int(rf.predict(row)[0])
    
    return {
        "placement_status": "Placed" if pred_class == 1 else "Not Placed",
        "prob_placed": round(prob_placed, 1),
        "prob_not_placed": round(100.0 - prob_placed, 1),
        "model_name": "Random Forest Ensemble",
        "inputs": inputs
    }


def save_random_forest_report(rf_data):
    """Save Random Forest & Bagging evaluation report to Output/Report/random_forest_report.txt."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    m = rf_data["metrics"]
    rep_path = config.REPORTS_DIR / "random_forest_report.txt"
    
    with open(rep_path, "w") as f:
        f.write("RANDOM FOREST & BAGGING REPORT (50,000 PLACEMENT RECORDS)\n")
        f.write("=" * 60 + "\n")
        f.write("Random Forest Classifier (100 Estimators, max_features='sqrt'):\n")
        f.write(f"  Test Accuracy:     {m['rf_accuracy']}%\n")
        f.write(f"  Out-of-Bag (OOB):  {m['rf_oob_score']}%\n")
        f.write(f"  ROC-AUC Score:     {m['rf_roc_auc']}%\n")
        f.write(f"  Precision:         {m['rf_precision']}%\n")
        f.write(f"  Recall:            {m['rf_recall']}%\n")
        f.write(f"  F1-Score:          {m['rf_f1']}%\n\n")
        f.write("Standard Bagging Comparison:\n")
        f.write(f"  Test Accuracy:     {m['bagging_accuracy']}%\n")
        f.write(f"  OOB Score:         {m['bagging_oob_score']}%\n\n")
        f.write("Top Feature Importances (Gini Impurity Reduction):\n")
        for item in m["top_features"]:
            f.write(f"  - {item['feature']:<20}: {item['importance']}%\n")
            
    return str(rep_path)


if __name__ == "__main__":
    print("=" * 70)
    print("  RANDOM FOREST & BAGGING - PLACEMENT PREDICTION")
    print("=" * 70)
    df = load_cleaned()
    rf_data = train_random_forest_classifier(df=df, n_estimators=100)
    print(f"Random Forest Accuracy: {rf_data['metrics']['rf_accuracy']}%")
    print(f"Out-of-Bag (OOB) Score: {rf_data['metrics']['rf_oob_score']}%")
    print(f"ROC-AUC Score: {rf_data['metrics']['rf_roc_auc']}%")
    print(f"Bagging Accuracy: {rf_data['metrics']['bagging_accuracy']}%")
    diagrams = generate_random_forest_diagrams(model_data=rf_data)
    print(f"Generated diagrams in: {config.PLOTS_DIR}")
    rep = save_random_forest_report(rf_data)
    print(f"Report saved: {rep}")
    print("=" * 70)
