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
        "model": binary_model,
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
    """Generate empirical S-curve and sigmoid fit diagram."""
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if model_data is None:
        model_data = train_logistic_regression(df)
        
    X_train = model_data["X_train"]
    y_train = model_data["y_train"]
    
    sns.set_theme(style="whitegrid")
    
    # -------------------------------------------------------------
    # DIAGRAM: Empirical CGPA vs Placement S-Curve & Sigmoid
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
    # DIAGRAM 2: Top Coefficients / Feature Impact Bar Chart
    # -------------------------------------------------------------
    coef_info = get_model_coefficients_and_odds_ratios(model_data=model_data)
    top15 = coef_info["table"].head(15).iloc[::-1]  # Reverse for horizontal bar chart
    
    fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=120)
    colors = ["#0f9f8f" if c > 0 else "#e05242" for c in top15["coefficient"]]
    ax.barh(top15["feature"], top15["coefficient"], color=colors, edgecolor="#21312f", linewidth=0.8, alpha=0.85)
    ax.axvline(0, color="#6f7f7a", linewidth=1.2, linestyle="--")
    ax.set_title("Top Features Driving Placement Probability (Standardized Logistic Regression)", fontsize=11, weight="bold", color="#21312f", pad=12)
    ax.set_xlabel("Standardized Coefficient (Positive = Increases Placement Odds, Negative = Decreases)", fontsize=9, weight="bold", color="#21312f")
    ax.set_ylabel("Feature", fontsize=9, weight="bold", color="#21312f")
    plt.tight_layout()
    plot2_path = config.PLOTS_DIR / "top_coefficients.png"
    fig.savefig(plot2_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    
    return {
        "plot_s_curve": "logistic_s_curve.png",
        "plot_top_coefficients": "top_coefficients.png"
    }


def get_model_coefficients_and_odds_ratios(model_data=None, df=None):
    """Compute standardized coefficients, Odds Ratios (exp(beta)), and multicollinearity diagnostics."""
    if model_data is None:
        model_data = train_logistic_regression(df)
        
    model = model_data.get("binary_model") or model_data.get("model")
    features = model_data["features"]
    
    coefs = model.coef_[0]
    odds_ratios = np.exp(coefs)
    
    table = pd.DataFrame({
        "feature": features,
        "coefficient": np.round(coefs, 3),
        "odds_ratio": np.round(odds_ratios, 3),
        "abs_impact": np.abs(coefs),
        "impact_direction": ["Increases Odds" if c > 0 else "Decreases Odds" for c in coefs],
    }).sort_values("abs_impact", ascending=False)
    
    # Multicollinearity Check: Check SGPA vs CGPA correlation
    X_train = model_data["X_train"]
    sgpa_cols = [c for c in features if c.startswith("SGPA_Sem")]
    avg_corr = 0.0
    if "CGPA" in X_train.columns and len(sgpa_cols) > 0:
        corr_series = X_train[["CGPA"] + sgpa_cols].corr()["CGPA"].drop("CGPA", errors="ignore")
        avg_corr = round(float(corr_series.mean()), 3)
        
    multicollinearity_note = {
        "average_cgpa_sgpa_correlation": avg_corr,
        "is_multicollinear": avg_corr > 0.7,
        "explanation": (
            f"CGPA correlates strongly ({avg_corr}) with the 8 SGPA semester columns. "
            "Because they represent nearly identical information, multicollinearity can cause individual "
            "coefficients to split or reverse sign (e.g. CGPA appearing slightly negative), even when overall accuracy is high."
        ) if avg_corr > 0.7 else "No severe multicollinearity detected between core academic indicators."
    }
    
    return {
        "table": table,
        "top_features": table.head(10).to_dict(orient="records"),
        "multicollinearity_note": multicollinearity_note
    }


def save_coefficients_report(model_data=None, df=None):
    """Save Logistic Regression Odds Ratios and coefficients report to Output/Report/coefficients_report.txt."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    info = get_model_coefficients_and_odds_ratios(model_data=model_data, df=df)
    rep_path = config.REPORTS_DIR / "coefficients_report.txt"
    
    with open(rep_path, "w") as f:
        f.write("LOGISTIC REGRESSION COEFFICIENTS & ODDS RATIOS REPORT (50,000 RECORDS)\n")
        f.write("=" * 70 + "\n")
        f.write("Standardized Coefficients & Multiplicative Odds Ratios:\n")
        f.write(f"{'Feature':<22} {'Coefficient':<14} {'Odds Ratio':<12} {'Direction'}\n")
        f.write("-" * 70 + "\n")
        for _, row in info["table"].head(15).iterrows():
            f.write(f"{row['feature']:<22} {row['coefficient']:>10.3f}   x{row['odds_ratio']:<10.3f}  {row['impact_direction']}\n")
            
        f.write("\nMulticollinearity Diagnostic:\n")
        f.write(f"  {info['multicollinearity_note']['explanation']}\n")
        f.write(f"\nGenerated Plot: top_coefficients.png\n")
        
    return str(rep_path)


if __name__ == "__main__":
    print("=" * 70)
    print("  LOGISTIC REGRESSION & COEFFICIENT INTERPRETATION")
    print("=" * 70)
    df = load_cleaned()
    m_data = train_logistic_regression(df=df)
    print(f"Validation Accuracy: {m_data['metrics']['accuracy']}% | ROC-AUC: {m_data['metrics']['auc']}")
    diagrams = generate_logistic_diagrams(model_data=m_data)
    rep = save_coefficients_report(model_data=m_data)
    print(f"Report saved to: {rep}")
    print("=" * 70)

