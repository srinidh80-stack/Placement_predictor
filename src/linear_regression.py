"""Linear regression model training, evaluation, plot generation, and salary package prediction (Module 2 Sessions 17-18)."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import config
from src.data_utils import clean_data, load_cleaned

REGRESSION_FEATURES = ["CGPA", "CodingTestScore", "MockInterviewScore", "AptitudeTestScore"]
TARGET_COL = "Salary Package"


def get_regression_dataset(df=None):
    """Filter placed students with valid features and target salary."""
    data = load_cleaned() if df is None else df.copy()
    
    if "PlacementStatus" in data.columns:
        placed = data[data["PlacementStatus"] == 1].copy()
    else:
        placed = data.copy()
        
    if TARGET_COL in placed.columns:
        placed = placed.dropna(subset=[TARGET_COL] + [f for f in REGRESSION_FEATURES if f in placed.columns])
    return placed


def run_batch_gradient_descent(X_train, y_train, alpha=0.01, epochs=300):
    """Numpy-based Batch Gradient Descent implementation as taught in Session 17-18."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    X_std = (X_train - mean) / std
    
    m = len(y_train)
    X_b = np.c_[np.ones((m, 1)), X_std]
    theta = np.zeros(X_b.shape[1])
    
    y_vec = y_train.to_numpy()
    cost_history = []
    
    for _ in range(epochs):
        y_hat = X_b.dot(theta)
        error = y_hat - y_vec
        gradients = (2.0 / m) * X_b.T.dot(error)
        theta -= alpha * gradients
        cost = (1.0 / m) * np.sum(error ** 2)
        cost_history.append(float(cost))
        
    return {
        "final_cost": round(cost_history[-1], 4),
        "initial_cost": round(cost_history[0], 4),
        "epochs": epochs,
        "learning_rate": alpha
    }


def train_regression_model(df=None, test_size=0.2, random_state=42):
    """Train Multiple Linear Regression and Single Feature (CGPA) baseline."""
    placed = get_regression_dataset(df)
    
    available_features = [f for f in REGRESSION_FEATURES if f in placed.columns]
    X = placed[available_features]
    y = placed[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Fit Sklearn Normal Equation model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_test_pred = model.predict(X_test)
    
    mse = float(mean_squared_error(y_test, y_test_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test, y_test_pred))
    r2 = float(r2_score(y_test, y_test_pred))
    
    # Baseline: CGPA only model for comparison (Session 17 Slide 24)
    cgpa_model = LinearRegression()
    cgpa_model.fit(X_train[["CGPA"]], y_train)
    cgpa_test_pred = cgpa_model.predict(X_test[["CGPA"]])
    cgpa_r2 = float(r2_score(y_test, cgpa_test_pred))
    cgpa_rmse = float(np.sqrt(mean_squared_error(y_test, cgpa_test_pred)))
    
    # Batch Gradient Descent stats
    gd_stats = run_batch_gradient_descent(X_train, y_train)
    
    # Coefficient breakdown
    coefficients = [
        {
            "feature": "Intercept (θ₀)",
            "weight": round(float(model.intercept_), 4),
            "description": "Baseline expected salary before adding feature adjustments",
            "impact": "Baseline constant",
        }
    ]
    
    for feat, weight in zip(available_features, model.coef_):
        coefficients.append({
            "feature": feat,
            "weight": round(float(weight), 4),
            "description": f"Slope / multiplier weight for {feat}",
            "impact": f"{'+' if weight > 0 else ''}{round(weight, 3)} LPA per point",
        })
        
    return {
        "model": model,
        "features": available_features,
        "X_test": X_test,
        "y_test": y_test,
        "y_test_pred": y_test_pred,
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
        "intercept": round(float(model.intercept_), 4),
        "coef_map": {feat: float(w) for feat, w in zip(available_features, model.coef_)},
        "coefficients": coefficients,
        "metrics": {
            "r2": round(r2, 4),
            "r2_percentage": round(r2 * 100, 1),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
        },
        "baseline_cgpa": {
            "r2": round(cgpa_r2, 4),
            "rmse": round(cgpa_rmse, 4),
            "gain_r2": round((r2 - cgpa_r2) * 100, 1),
        },
        "gd_stats": gd_stats,
    }


def calculate_salary_prediction(cgpa, coding_score, interview_score, aptitude_score, model_data=None):
    """Predict salary package (LPA) given input scores using the trained linear model."""
    if model_data is None:
        model_data = train_regression_model()
        
    intercept = model_data["intercept"]
    coefs = model_data["coef_map"]
    
    cgpa_contrib = coefs.get("CGPA", 4.30) * cgpa
    coding_contrib = coefs.get("CodingTestScore", -0.048) * coding_score
    interview_contrib = coefs.get("MockInterviewScore", 0.115) * interview_score
    aptitude_contrib = coefs.get("AptitudeTestScore", -0.062) * aptitude_score
    
    raw_prediction = intercept + cgpa_contrib + coding_contrib + interview_contrib + aptitude_contrib
    predicted_salary = max(3.0, min(26.0, float(raw_prediction)))
    
    return {
        "predicted_salary": round(predicted_salary, 2),
        "raw_prediction": round(raw_prediction, 2),
        "inputs": {
            "cgpa": cgpa,
            "coding_score": coding_score,
            "interview_score": interview_score,
            "aptitude_score": aptitude_score,
        },
        "breakdown": [
            {"component": "Intercept (θ₀)", "value": round(intercept, 2)},
            {"component": "CGPA Contribution", "value": round(cgpa_contrib, 2)},
            {"component": "Coding Score Contribution", "value": round(coding_contrib, 2)},
            {"component": "Mock Interview Contribution", "value": round(interview_contrib, 2)},
            {"component": "Aptitude Score Contribution", "value": round(aptitude_contrib, 2)},
        ]
    }


def generate_regression_diagrams(df=None, model_data=None, current_prediction=None):
    """Generate visual scatter plot with regression line & clusters diagram and actual vs predicted plot."""
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    placed = get_regression_dataset(df)
    
    if model_data is None:
        model_data = train_regression_model(placed)
    model = model_data["model"]
    
    sns.set_theme(style="whitegrid")
    
    # -------------------------------------------------------------
    # DIAGRAM 1: CGPA vs Salary Package Scatter + Regression Line & Clusters
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=120)
    sample_df = placed.sample(n=min(2500, len(placed)), random_state=42)
    
    ax.scatter(sample_df["CGPA"], sample_df["Salary Package"], alpha=0.30, color="#0f9f8f", s=18, label="Actual Placed Students")
    
    cgpa_grid = np.linspace(sample_df["CGPA"].min(), sample_df["CGPA"].max(), 200)
    mean_coding = sample_df["CodingTestScore"].mean()
    mean_interview = sample_df["MockInterviewScore"].mean()
    mean_aptitude = sample_df["AptitudeTestScore"].mean()
    
    X_grid = pd.DataFrame({
        "CGPA": cgpa_grid,
        "CodingTestScore": mean_coding,
        "MockInterviewScore": mean_interview,
        "AptitudeTestScore": mean_aptitude
    })
    y_line = model.predict(X_grid)
    
    ax.plot(cgpa_grid, y_line, color="#f05f4f", linewidth=2.8, label=f"Fitted Linear Model (R² = {model_data['metrics']['r2']})")
    ax.axvline(x=8.0, color="#6f7f7a", linestyle="--", alpha=0.75, label="CGPA 8.0 Tier Cutoff")
    
    # Annotate Clusters and Empty Gap
    ax.text(6.3, 11.5, "Tier-2 Cluster (~8-9 LPA)\n(49.7% of students)", fontsize=8.5, color="#0a4f4b", weight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f8f4", edgecolor="#0f9f8f", alpha=0.9))
    
    ax.text(8.3, 23.5, "Tier-1 Cluster (~19-21 LPA)\n(50.3% of students)", fontsize=8.5, color="#0a4f4b", weight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f8f4", edgecolor="#0f9f8f", alpha=0.9))
            
    ax.text(6.0, 15.2, "← Straight line passes through empty gap (13-18 LPA) →", fontsize=8.5, color="#d84f40", style="italic", weight="bold")

    # If current user prediction exists, mark it on the plot!
    if current_prediction:
        input_cgpa = current_prediction["inputs"]["cgpa"]
        pred_sal = current_prediction["predicted_salary"]
        ax.scatter([input_cgpa], [pred_sal], color="#f5b84b", s=140, edgecolor="#21312f", linewidth=2, zorder=10, label=f"Current Prediction ({input_cgpa} CGPA → {pred_sal} LPA)")
        ax.annotate(f"Prediction: {pred_sal} LPA", xy=(input_cgpa, pred_sal), xytext=(input_cgpa - 1.1, pred_sal + 2.5),
                    arrowprops=dict(facecolor="#21312f", shrink=0.08, width=1.5, headwidth=6),
                    fontsize=9, weight="bold", color="#21312f", bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff8ea", edgecolor="#f5b84b"))

    ax.set_title("CGPA vs Salary Package with Linear Regression Line & Cluster Gap", fontsize=11, weight="bold", color="#21312f", pad=12)
    ax.set_xlabel("CGPA (Cumulative Grade Point Average)", fontsize=9.5, weight="bold", color="#21312f")
    ax.set_ylabel("Salary Package (LPA)", fontsize=9.5, weight="bold", color="#21312f")
    ax.set_ylim(0, 30)
    ax.set_xlim(5.0, 10.0)
    ax.legend(frameon=True, loc="lower right", fontsize=8)
    plt.tight_layout()
    plot1_path = config.PLOTS_DIR / "regression_line_fit.png"
    fig.savefig(plot1_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    
    # -------------------------------------------------------------
    # DIAGRAM 2: Actual vs Predicted Salary on Validation Test Set
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=120)
    y_test = model_data["y_test"]
    y_pred = model_data["y_test_pred"]
    
    sample_idx = np.random.choice(len(y_test), size=min(1500, len(y_test)), replace=False)
    ax.scatter(y_test.iloc[sample_idx], y_pred[sample_idx], alpha=0.35, color="#0b8276", s=18, label="Test Predictions")
    
    # 45-degree reference line
    lims = [0, 30]
    ax.plot(lims, lims, color="#f05f4f", linestyle="--", linewidth=2, label="Perfect 1:1 Fit (y = ŷ)")
    
    ax.set_title(f"Actual vs Predicted Salary on Test Set (RMSE = {model_data['metrics']['rmse']} LPA, R² = {model_data['metrics']['r2']})", fontsize=11, weight="bold", color="#21312f", pad=12)
    ax.set_xlabel("Actual Salary Package (LPA)", fontsize=9.5, weight="bold", color="#21312f")
    ax.set_ylabel("Predicted Salary Package (LPA)", fontsize=9.5, weight="bold", color="#21312f")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)
    ax.legend(frameon=True, loc="upper left", fontsize=8.5)
    plt.tight_layout()
    plot2_path = config.PLOTS_DIR / "regression_actual_vs_predicted.png"
    fig.savefig(plot2_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    
    return {
        "plot_line_fit": "plots/regression_line_fit.png",
        "plot_actual_vs_pred": "plots/regression_actual_vs_predicted.png"
    }
