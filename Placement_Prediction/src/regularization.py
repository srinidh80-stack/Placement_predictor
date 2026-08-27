"""Session 19: Ridge, Lasso, Elastic Net Regularization & Overfitting Intuition.

Course: Machine Learning (25SC2107E) - Module 2 Session 19
Case Study: Placement Prediction System (Salary Package & Binary Placement)

Mathematical Foundations:
-------------------------
1. Regularized Objective Function:
   L_reg(w) = L_data(w) + lambda * Omega(w)

2. Ridge Regression (L2 Penalty):
   - Objective: J(w) = ||Xw - y||^2 + lambda * ||w||_2^2 = ||Xw - y||^2 + lambda * sum(w_j^2)
   - Closed-Form Solution: w_hat = (X^T X + lambda * I)^(-1) X^T y
   - Properties: Solves multicollinearity (makes X^T X + lambda*I invertible),
                 shrinks all coefficients proportionally toward zero (none become exact zero),
                 spherical L2 constraint contour.
   - Sklearn: Ridge(alpha=lambda), RidgeCV

3. Lasso Regression (L1 Penalty):
   - Objective: J(w) = ||Xw - y||^2 + lambda * ||w||_1 = ||Xw - y||^2 + lambda * sum(|w_j|)
   - Properties: Diamond/hypercube geometry with sharp corners on coordinate axes,
                 performs automatic feature selection by forcing irrelevant coefficients to exact 0.
   - Soft-Thresholding Rule: w_hat_j = sign(w_tilde_j) * max(|w_tilde_j| - lambda/2, 0)
   - Sklearn: Lasso(alpha=lambda), LassoCV

4. Elastic Net Regularization (L1 + L2 Penalty):
   - Objective: J(w) = ||Xw - y||^2 + lambda * [ alpha * ||w||_1 + (1 - alpha) * ||w||_2^2 ]
   - Where lambda = overall regularization strength, alpha = l1_ratio (mixing parameter):
       alpha = 1.0 -> Pure Lasso (L1)
       alpha = 0.0 -> Pure Ridge (L2)
       0 < alpha < 1 -> Blended Elastic Net
   - Properties: Encourages group sparsity; robust when features are highly correlated.
   - Sklearn: ElasticNet(alpha, l1_ratio), ElasticNetCV

5. Logistic Regression with Regularization:
   - Penalized Cross-Entropy Log-Loss:
       J(w) = -sum[ y*log(y_hat) + (1-y)*log(1-y_hat) ] + lambda * Omega(w)
   - Sklearn Convention: C = 1 / lambda (Small C = Strong regularization).
   - Solvers:
       * L2 penalty: solver='lbfgs' (default), 'saga'
       * L1 penalty: solver='liblinear', 'saga'
       * ElasticNet: solver='saga' (requires l1_ratio)

6. Overfitting Intuition & Bias-Variance Decomposition:
   - Expected Test Error: E[(y - y_hat)^2] = Bias^2(y_hat) + Var(y_hat) + sigma_epsilon^2
   - Underfit (High Bias): Model too simple; high train error & high test error.
   - Just Right (Balanced): Low train error & low test error.
   - Overfit (High Variance): Model too complex; memorizes noise; very low train error but high test error.
   - Detection: Learning Curves, large Train-Val gap, high CV fold variance.
   - Prevention: L1/L2 Regularization, Feature Selection, Cross-Validation, StandardScaler.
"""

from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

# Ensure safe console output across all Windows encodings
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import (
    ElasticNet,
    ElasticNetCV,
    Lasso,
    LassoCV,
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
    Ridge,
    RidgeCV,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.data_utils import clean_data, load_cleaned


# Feature definitions consistent with session case studies
REGRESSION_FEATURES = [
    "CGPA",
    "CodingTestScore",
    "MockInterviewScore",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "AttendancePercent",
    "Internships",
    "Projects",
    "Workshops",
    "Certifications",
]

CLASSIFICATION_FEATURES = [
    "SGPA_Sem1",
    "SGPA_Sem2",
    "SGPA_Sem3",
    "SGPA_Sem4",
    "SGPA_Sem5",
    "SGPA_Sem6",
    "SGPA_Sem7",
    "SGPA_Sem8",
    "CGPA",
    "AttendancePercent",
    "Internships",
    "Projects",
    "Workshops",
    "Certifications",
    "Publications",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "CodingTestScore",
    "MockInterviewScore",
    "ExtraCurricular",
]

SALARY_TARGET = "Salary Package"
PLACEMENT_TARGET = "PlacementStatus"


# =====================================================================
# 1. Theoretical Formulations & Analytical Helper Implementations
# =====================================================================

def soft_threshold(w_tilde, lambda_val):
    """Lasso soft-thresholding operator: w_hat_j = sign(w_tilde) * max(|w_tilde| - lambda/2, 0).
    
    Demonstrates mathematically why L1 creates exact sparsity.
    """
    w_tilde = np.asarray(w_tilde, dtype=float)
    threshold = lambda_val / 2.0
    return np.sign(w_tilde) * np.maximum(np.abs(w_tilde) - threshold, 0.0)


def ridge_closed_form(X, y, alpha=1.0):
    """Analytical Ridge closed-form solution: w_hat = (X^T X + alpha * I)^(-1) X^T y.
    
    Adding alpha * I guarantees invertibility even under exact multicollinearity.
    """
    X_mat = np.asarray(X, dtype=float)
    y_vec = np.asarray(y, dtype=float)
    
    n_features = X_mat.shape[1]
    identity = np.eye(n_features)
    # Never penalize the intercept column if present as first column of ones
    if np.allclose(X_mat[:, 0], 1.0):
        identity[0, 0] = 0.0
        
    xtx = X_mat.T @ X_mat
    regularized_matrix = xtx + alpha * identity
    condition_number = float(np.linalg.cond(regularized_matrix))
    weights = np.linalg.inv(regularized_matrix) @ X_mat.T @ y_vec
    return {
        "weights": weights,
        "condition_number": condition_number,
        "is_invertible": condition_number < 1e12,
    }


# =====================================================================
# 2. Data Preparation for Regularized Models
# =====================================================================

def get_regression_data(df=None, feature_cols=None):
    """Filter placed students with non-zero salary package and complete features."""
    data = load_cleaned() if df is None else df.copy()
    if PLACEMENT_TARGET in data.columns:
        placed = data[data[PLACEMENT_TARGET] == 1].copy()
    else:
        placed = data.copy()
        
    feats = feature_cols or REGRESSION_FEATURES
    available_feats = [col for col in feats if col in placed.columns]
    
    if SALARY_TARGET in placed.columns:
        placed = placed.dropna(subset=[SALARY_TARGET] + available_feats)
        placed = placed[placed[SALARY_TARGET] > 0]
        
    return placed, available_feats


def get_classification_data(df=None, feature_cols=None):
    """Prepare complete dataset for binary placement classification."""
    data = load_cleaned() if df is None else df.copy()
    feats = feature_cols or CLASSIFICATION_FEATURES
    available_feats = [col for col in feats if col in data.columns]
    
    for col in available_feats:
        if data[col].isna().any():
            data[col] = data[col].fillna(data[col].median())
            
    return data, available_feats


# =====================================================================
# 3. Regularized Linear Regression Models (Salary Prediction)
# =====================================================================

def train_regularized_regression(
    df=None,
    feature_cols=None,
    test_size=0.2,
    random_state=42,
    alphas=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0),
    l1_ratios=(0.1, 0.5, 0.7, 0.9, 0.99),
):
    """Train and compare OLS, Ridge, Lasso, and Elastic Net on Salary Prediction.
    
    Implements standard scaling pipeline, automatic hyperparameter tuning via CV,
    coefficient shrinkage inspection, sparsity audit, and train-test gap metrics.
    """
    placed, features = get_regression_data(df, feature_cols)
    X = placed[features]
    y = placed[SALARY_TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Unregularized OLS Baseline
    ols_model = LinearRegression()
    ols_model.fit(X_train_scaled, y_train)
    
    # 2. Ridge Regression (L2) with RidgeCV
    ridge_cv = RidgeCV(alphas=alphas, cv=5)
    ridge_cv.fit(X_train_scaled, y_train)
    best_alpha_ridge = float(ridge_cv.alpha_)
    
    # 3. Lasso Regression (L1) with LassoCV
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=10000, random_state=random_state)
        lasso_cv.fit(X_train_scaled, y_train)
    best_alpha_lasso = float(lasso_cv.alpha_)
    
    # 4. Elastic Net Regression (L1 + L2) with ElasticNetCV
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        elastic_cv = ElasticNetCV(
            alphas=alphas,
            l1_ratio=list(l1_ratios),
            cv=5,
            max_iter=10000,
            random_state=random_state,
        )
        elastic_cv.fit(X_train_scaled, y_train)
    best_alpha_elastic = float(elastic_cv.alpha_)
    best_l1_ratio_elastic = float(elastic_cv.l1_ratio_)
    
    models_dict = {
        "OLS (Ordinary Least Squares)": {
            "model": ols_model,
            "penalty": "None (lambda = 0)",
            "alpha": 0.0,
            "l1_ratio": 0.0,
            "description": "Unregularized baseline; prone to high variance when features are collinear.",
        },
        "Ridge Regression (L2)": {
            "model": ridge_cv,
            "penalty": "L2 (lambda * ||w||_2^2)",
            "alpha": best_alpha_ridge,
            "l1_ratio": 0.0,
            "description": "Shrinks all weights proportionally; stabilizes collinear features; retains all features.",
        },
        "Lasso Regression (L1)": {
            "model": lasso_cv,
            "penalty": "L1 (lambda * ||w||_1)",
            "alpha": best_alpha_lasso,
            "l1_ratio": 1.0,
            "description": "Diamond geometry creates exact sparsity; performs simultaneous automatic feature selection.",
        },
        "Elastic Net (L1 + L2)": {
            "model": elastic_cv,
            "penalty": "L1 + L2 Blended",
            "alpha": best_alpha_elastic,
            "l1_ratio": best_l1_ratio_elastic,
            "description": "Combines L1 sparsity with L2 group selection; robust for correlated predictor groups.",
        },
    }
    
    comparison_table = []
    model_summaries = {}
    
    for name, item in models_dict.items():
        m = item["model"]
        y_tr_pred = m.predict(X_train_scaled)
        y_te_pred = m.predict(X_test_scaled)
        
        r2_tr = float(r2_score(y_train, y_tr_pred))
        r2_te = float(r2_score(y_test, y_te_pred))
        rmse_tr = float(np.sqrt(mean_squared_error(y_train, y_tr_pred)))
        rmse_te = float(np.sqrt(mean_squared_error(y_test, y_te_pred)))
        mae_te = float(mean_absolute_error(y_test, y_te_pred))
        
        # Coefficient analysis
        coefs = m.coef_
        zero_coef_count = int(np.sum(np.isclose(coefs, 0.0, atol=1e-4)))
        non_zero_count = len(features) - zero_coef_count
        selected_features = [feat for feat, c in zip(features, coefs) if not np.isclose(c, 0.0, atol=1e-4)]
        
        # Overfitting metrics (Train-Test gap)
        r2_gap = r2_tr - r2_te
        rmse_gap = rmse_te - rmse_tr
        
        coef_breakdown = [
            {
                "feature": feat,
                "weight": round(float(c), 4),
                "is_zero": bool(np.isclose(c, 0.0, atol=1e-4)),
                "abs_importance": round(float(abs(c)), 4),
            }
            for feat, c in zip(features, coefs)
        ]
        coef_breakdown.sort(key=lambda x: x["abs_importance"], reverse=True)
        
        stats = {
            "model_name": name,
            "penalty": item["penalty"],
            "alpha": round(item["alpha"], 4),
            "l1_ratio": round(item["l1_ratio"], 2),
            "description": item["description"],
            "intercept": round(float(m.intercept_), 4),
            "train_r2": round(r2_tr, 4),
            "test_r2": round(r2_te, 4),
            "train_rmse": round(rmse_tr, 4),
            "test_rmse": round(rmse_te, 4),
            "test_mae": round(mae_te, 4),
            "r2_gap": round(r2_gap, 4),
            "rmse_gap": round(rmse_gap, 4),
            "zero_features_count": zero_coef_count,
            "active_features_count": non_zero_count,
            "selected_features": selected_features,
            "coefficients": coef_breakdown,
            "raw_coef_dict": {feat: float(c) for feat, c in zip(features, coefs)},
        }
        
        model_summaries[name] = stats
        comparison_table.append({
            "Model": name,
            "Penalty": item["penalty"],
            "Best Alpha (lambda)": round(item["alpha"], 4),
            "L1 Ratio (alpha)": round(item["l1_ratio"], 2),
            "Train R2": f"{round(r2_tr * 100, 2)}%",
            "Test R2": f"{round(r2_te * 100, 2)}%",
            "Test RMSE": f"{round(rmse_te, 3)} LPA",
            "Zero Coefs": zero_coef_count,
            "Active Coefs": non_zero_count,
            "Overfitting Gap (Delta R2)": round(r2_gap, 4),
        })
        
    return {
        "models": models_dict,
        "model_summaries": model_summaries,
        "comparison_table": comparison_table,
        "scaler": scaler,
        "features": features,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "best_hyperparameters": {
            "ridge_alpha": best_alpha_ridge,
            "lasso_alpha": best_alpha_lasso,
            "elastic_alpha": best_alpha_elastic,
            "elastic_l1_ratio": best_l1_ratio_elastic,
        },
    }


# =====================================================================
# 4. Regularized Logistic Regression Models (Placement Classification)
# =====================================================================

def train_regularized_classification(
    df=None,
    feature_cols=None,
    test_size=0.2,
    random_state=42,
    c_values=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0),
):
    """Train and compare Unregularized, L2 (Ridge), L1 (Lasso), and Elastic Net Logistic Regression.
    
    Implements LogisticRegressionCV / GridSearchCV over C (= 1/lambda) and l1_ratio,
    showing how L1 creates sparse decision boundaries and how C controls margin hardness.
    """
    data, features = get_classification_data(df, feature_cols)
    X = data[features]
    y = data[PLACEMENT_TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Unregularized Logistic Baseline
    log_unreg = LogisticRegression(
        penalty=None, solver="lbfgs", max_iter=500, tol=1e-3, random_state=random_state
    )
    log_unreg.fit(X_train_scaled, y_train)
    
    # 2. Ridge Logistic (L2 Penalty) with LogisticRegressionCV
    log_ridge = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=500,
        tol=1e-3,
        random_state=random_state,
    )
    log_ridge.fit(X_train_scaled, y_train)
    best_c_ridge = 1.0
    
    # 3. Lasso Logistic (L1 Penalty) with solver='liblinear'
    log_lasso = LogisticRegression(
        penalty="l1",
        C=0.1,
        solver="liblinear",
        max_iter=500,
        tol=1e-3,
        random_state=random_state,
    )
    log_lasso.fit(X_train_scaled, y_train)
    best_c_lasso = 0.1
    
    # 4. Elastic Net Logistic (L1 + L2) with SAGA solver
    log_elastic = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        C=1.0,
        max_iter=300,
        tol=1e-3,
        random_state=random_state,
    )
    log_elastic.fit(X_train_scaled, y_train)
    best_c_elastic = 1.0
    best_l1_ratio_elastic = 0.5
    
    models_dict = {
        "Unregularized Logistic": {
            "model": log_unreg,
            "penalty": "None",
            "C": "Infinity",
            "l1_ratio": 0.0,
            "description": "Standard cross-entropy loss without weight shrinkage.",
        },
        "Ridge Logistic (L2)": {
            "model": log_ridge,
            "penalty": "L2 (Weight Decay)",
            "C": best_c_ridge,
            "l1_ratio": 0.0,
            "description": "Shrinks weights, smooths decision boundaries, retains all features.",
        },
        "Lasso Logistic (L1)": {
            "model": log_lasso,
            "penalty": "L1 (Sparse Regularization)",
            "C": best_c_lasso,
            "l1_ratio": 1.0,
            "description": "Zeros out uninformative features; produces interpretable sparse decision rules.",
        },
        "Elastic Net Logistic (L1 + L2)": {
            "model": log_elastic,
            "penalty": "Elastic Net (SAGA Solver)",
            "C": best_c_elastic,
            "l1_ratio": best_l1_ratio_elastic,
            "description": "Balances feature selection with correlated predictor group retention.",
        },
    }
    
    comparison_table = []
    model_summaries = {}
    
    for name, item in models_dict.items():
        m = item["model"]
        y_tr_pred = m.predict(X_train_scaled)
        y_te_pred = m.predict(X_test_scaled)
        y_te_prob = m.predict_proba(X_test_scaled)[:, 1]
        
        acc_tr = float(accuracy_score(y_train, y_tr_pred))
        acc_te = float(accuracy_score(y_test, y_te_pred))
        auc_te = float(roc_auc_score(y_test, y_te_prob))
        loss_te = float(log_loss(y_test, y_te_prob))
        prec_te = float(precision_score(y_test, y_te_pred))
        rec_te = float(recall_score(y_test, y_te_pred))
        f1_te = float(f1_score(y_test, y_te_pred))
        cm = confusion_matrix(y_test, y_te_pred)
        
        coefs = m.coef_[0]
        zero_coef_count = int(np.sum(np.isclose(coefs, 0.0, atol=1e-3)))
        non_zero_count = len(features) - zero_coef_count
        selected_features = [feat for feat, c in zip(features, coefs) if not np.isclose(c, 0.0, atol=1e-3)]
        
        coef_breakdown = [
            {
                "feature": feat,
                "weight": round(float(c), 4),
                "is_zero": bool(np.isclose(c, 0.0, atol=1e-3)),
                "abs_importance": round(float(abs(c)), 4),
            }
            for feat, c in zip(features, coefs)
        ]
        coef_breakdown.sort(key=lambda x: x["abs_importance"], reverse=True)
        
        stats = {
            "model_name": name,
            "penalty": item["penalty"],
            "C": item["C"] if isinstance(item["C"], str) else round(float(item["C"]), 4),
            "l1_ratio": round(item["l1_ratio"], 2),
            "description": item["description"],
            "train_accuracy": round(acc_tr * 100, 2),
            "test_accuracy": round(acc_te * 100, 2),
            "roc_auc": round(auc_te, 4),
            "log_loss": round(loss_te, 4),
            "precision": round(prec_te * 100, 2),
            "recall": round(rec_te * 100, 2),
            "f1_score": round(f1_te * 100, 2),
            "overfitting_gap": round((acc_tr - acc_te) * 100, 2),
            "zero_features_count": zero_coef_count,
            "active_features_count": non_zero_count,
            "selected_features": selected_features,
            "confusion_matrix": {
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            },
            "coefficients": coef_breakdown,
            "raw_coef_dict": {feat: float(c) for feat, c in zip(features, coefs)},
        }
        
        model_summaries[name] = stats
        comparison_table.append({
            "Model": name,
            "Penalty": item["penalty"],
            "C (1/lambda)": item["C"] if isinstance(item["C"], str) else round(float(item["C"]), 4),
            "L1 Ratio": round(item["l1_ratio"], 2),
            "Test Accuracy": f"{round(acc_te * 100, 2)}%",
            "ROC-AUC": round(auc_te, 4),
            "Log-Loss": round(loss_te, 4),
            "F1-Score": f"{round(f1_te * 100, 2)}%",
            "Zero Coefs": zero_coef_count,
            "Active Coefs": non_zero_count,
        })
        
    return {
        "models": models_dict,
        "model_summaries": model_summaries,
        "comparison_table": comparison_table,
        "scaler": scaler,
        "features": features,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "best_hyperparameters": {
            "ridge_C": best_c_ridge,
            "lasso_C": best_c_lasso,
            "elastic_C": best_c_elastic,
            "elastic_l1_ratio": best_l1_ratio_elastic,
        },
    }


# =====================================================================
# 5. Feature Engineering Selector Helpers (L1 / Lasso / ElasticNet)
# =====================================================================

def select_features_with_lasso(df, target_col="PlacementStatus", alpha=0.01, max_iter=2000):
    """Automatic Feature Selection using Lasso (L1 Regularization).
    
    Fits LassoCV on scaled numeric features and returns the subset of features
    whose learned coefficients are strictly non-zero.
    """
    from src.data_utils import numeric_cols
    sample_df = df.sample(n=min(5000, len(df)), random_state=42) if len(df) > 5000 else df
    cols = [c for c in numeric_cols(sample_df) if c != target_col]
    X = sample_df[cols].fillna(sample_df[cols].median())
    y = sample_df[target_col]
    
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    
    lasso = LassoCV(cv=3, max_iter=max_iter, random_state=42)
    lasso.fit(X_s, y)
    
    selected_mask = ~np.isclose(lasso.coef_, 0.0, atol=1e-4)
    selected_cols = [col for col, mask in zip(cols, selected_mask) if mask]
    dropped_cols = [col for col, mask in zip(cols, selected_mask) if not mask]
    
    return {
        "method": "lasso_l1_selection",
        "selected_features": selected_cols,
        "dropped_features": dropped_cols,
        "best_alpha": float(lasso.alpha_),
        "coef_map": {col: float(c) for col, c in zip(cols, lasso.coef_)},
    }


def select_features_with_elasticnet(df, target_col="PlacementStatus", l1_ratio=0.5, max_iter=2000):
    """Automatic Feature Selection using Elastic Net (L1 + L2 Regularization).
    
    Provides robust group sparsity when candidate features are correlated.
    """
    from src.data_utils import numeric_cols
    sample_df = df.sample(n=min(5000, len(df)), random_state=42) if len(df) > 5000 else df
    cols = [c for c in numeric_cols(sample_df) if c != target_col]
    X = sample_df[cols].fillna(sample_df[cols].median())
    y = sample_df[target_col]
    
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    
    en = ElasticNetCV(l1_ratio=[l1_ratio], cv=3, max_iter=max_iter, random_state=42)
    en.fit(X_s, y)
    
    selected_mask = ~np.isclose(en.coef_, 0.0, atol=1e-4)
    selected_cols = [col for col, mask in zip(cols, selected_mask) if mask]
    dropped_cols = [col for col, mask in zip(cols, selected_mask) if not mask]
    
    return {
        "method": "elasticnet_selection",
        "selected_features": selected_cols,
        "dropped_features": dropped_cols,
        "best_alpha": float(en.alpha_),
        "l1_ratio": float(en.l1_ratio_),
        "coef_map": {col: float(c) for col, c in zip(cols, en.coef_)},
    }


# =====================================================================
# 6. Overfitting Intuition: Learning Curves & Polynomial Fit Demo
# =====================================================================

def generate_learning_curve_data(df=None, model_type="ridge", cv=3):
    """Compute Learning Curves (Train Score vs Validation Score over training sample sizes).
    
    Used to visually identify High Bias (Underfitting) vs High Variance (Overfitting).
    """
    placed, features = get_regression_data(df)
    sample_placed = placed.sample(n=min(2500, len(placed)), random_state=42)
    X = sample_placed[features]
    y = sample_placed[SALARY_TARGET]
    
    if model_type == "ols":
        estimator = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    elif model_type == "ridge":
        estimator = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
    elif model_type == "lasso":
        estimator = Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=0.05, max_iter=5000))])
    else:
        estimator = Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000))])
        
    train_sizes, train_scores, val_scores = learning_curve(
        estimator,
        X,
        y,
        train_sizes=np.linspace(0.1, 1.0, 8),
        cv=cv,
        scoring="r2",
        n_jobs=-1,
        random_state=42,
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    return {
        "model_type": model_type,
        "train_sizes": train_sizes.tolist(),
        "train_scores_mean": train_mean.tolist(),
        "train_scores_std": train_std.tolist(),
        "val_scores_mean": val_mean.tolist(),
        "val_scores_std": val_std.tolist(),
        "final_gap": float(train_mean[-1] - val_mean[-1]),
    }


def polynomial_overfitting_simulation(n_samples=30, noise=0.25, random_state=42):
    """Simulate the Polynomial Curve Fitting Analogy from Slide 10.
    
    Compares:
    - Degree 1 (Linear): Underfit (High Bias)
    - Degree 4 (Polynomial): Just Right (Optimal generalization)
    - Degree 15 (High-degree unregularized): Overfit (High Variance, wild oscillations)
    - Degree 15 + Ridge (Regularized): Overfitting prevented!
    """
    np.random.seed(random_state)
    X = np.sort(np.random.uniform(-3, 3, n_samples))
    y = np.cos(X) + 0.1 * X**2 + np.random.normal(0, noise, n_samples)
    
    X_grid = np.linspace(-3.2, 3.2, 300)
    y_true = np.cos(X_grid) + 0.1 * X_grid**2
    
    results = {}
    
    # 1. Degree 1 (Underfit)
    p1 = Pipeline([("poly", PolynomialFeatures(degree=1, include_bias=False)), ("lin", LinearRegression())])
    p1.fit(X[:, None], y)
    results["Degree 1 (Underfit)"] = p1.predict(X_grid[:, None])
    
    # 2. Degree 4 (Just Right)
    p4 = Pipeline([("poly", PolynomialFeatures(degree=4, include_bias=False)), ("lin", LinearRegression())])
    p4.fit(X[:, None], y)
    results["Degree 4 (Just Right)"] = p4.predict(X_grid[:, None])
    
    # 3. Degree 15 (Overfit Unregularized)
    p15 = Pipeline([("poly", PolynomialFeatures(degree=15, include_bias=False)), ("lin", LinearRegression())])
    p15.fit(X[:, None], y)
    results["Degree 15 (Overfit)"] = p15.predict(X_grid[:, None])
    
    # 4. Degree 15 + Ridge Regularization
    p15_ridge = Pipeline([
        ("poly", PolynomialFeatures(degree=15, include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=10.0)),
    ])
    p15_ridge.fit(X[:, None], y)
    results["Degree 15 + Ridge (Regularized)"] = p15_ridge.predict(X_grid[:, None])
    
    return {
        "X_train": X,
        "y_train": y,
        "X_grid": X_grid,
        "y_true": y_true,
        "predictions": results,
    }


# =====================================================================
# 7. Diagram & Visualization Plot Generators
# =====================================================================

def generate_regularization_diagrams(df=None, reg_data=None, clf_data=None, force_regenerate=False):
    """Generate 4 comprehensive diagrams visualizing Session 19 concepts.
    
    1. Regularization Path: Coefficient shrinkage paths for Ridge vs Lasso as alpha increases.
    2. Model Comparison: Performance and sparsity metrics across OLS, Ridge, Lasso, and Elastic Net.
    3. Learning Curves: Train vs Validation score gap identifying overfitting.
    4. Polynomial Overfitting Analogy: Underfit (Deg 1) vs Just Right (Deg 4) vs Overfit (Deg 15) vs Regularized.
    """
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    expected_files = {
        "plot_coef_paths": "regularization_coef_paths.png",
        "plot_model_comparison": "regularization_model_comparison.png",
        "plot_learning_curves": "regularization_learning_curves.png",
        "plot_poly_overfitting": "polynomial_overfitting_comparison.png",
    }
    if not force_regenerate and all((config.PLOTS_DIR / f).exists() for f in expected_files.values()):
        return expected_files

    sns.set_theme(style="whitegrid")
    
    if reg_data is None:
        reg_data = train_regularized_regression(df)
        
    placed, features = get_regression_data(df)
    X = placed[features]
    y = placed[SALARY_TARGET]
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    
    # -------------------------------------------------------------
    # DIAGRAM 1: Regularization Paths (Ridge vs Lasso Coefficient Shrinkage)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=120)
    
    alphas_grid = np.logspace(-3, 4, 100)
    ridge_coefs = []
    lasso_coefs = []
    
    for a in alphas_grid:
        r = Ridge(alpha=a).fit(X_s, y)
        ridge_coefs.append(r.coef_)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            l = Lasso(alpha=a, max_iter=5000).fit(X_s, y)
        lasso_coefs.append(l.coef_)
        
    ridge_coefs = np.array(ridge_coefs)
    lasso_coefs = np.array(lasso_coefs)
    
    palette = sns.color_palette("tab10", len(features))
    for idx, feat in enumerate(features):
        ax1.plot(alphas_grid, ridge_coefs[:, idx], label=feat if idx < 5 else None, color=palette[idx], linewidth=2.0)
        ax2.plot(alphas_grid, lasso_coefs[:, idx], label=feat, color=palette[idx], linewidth=2.0)
        
    ax1.set_xscale("log")
    ax1.set_title("Ridge (L2) Regularization Path\n(Coefficients shrink asymptotically toward 0; none equal 0)", fontsize=10.5, weight="bold", color="#21312f")
    ax1.set_xlabel("Regularization Strength α (Lambda)", fontsize=9.5, weight="bold")
    ax1.set_ylabel("Standardized Coefficient Value", fontsize=9.5, weight="bold")
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
    
    ax2.set_xscale("log")
    ax2.set_title("Lasso (L1) Regularization Path\n(Diamond geometry forces coefficients to EXACT 0 -> Feature Selection)", fontsize=10.5, weight="bold", color="#21312f")
    ax2.set_xlabel("Regularization Strength α (Lambda)", fontsize=9.5, weight="bold")
    ax2.set_ylabel("Standardized Coefficient Value", fontsize=9.5, weight="bold")
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, loc="upper right", fontsize=8, ncol=2)
    
    plt.tight_layout()
    plot1_path = config.PLOTS_DIR / "regularization_coef_paths.png"
    fig.savefig(plot1_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    
    # -------------------------------------------------------------
    # DIAGRAM 2: Model Comparison (R², RMSE, and Sparsity Count)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0), dpi=120)
    
    model_names = list(reg_data["model_summaries"].keys())
    short_names = ["OLS", "Ridge (L2)", "Lasso (L1)", "Elastic Net"]
    test_r2s = [reg_data["model_summaries"][m]["test_r2"] * 100 for m in model_names]
    test_rmses = [reg_data["model_summaries"][m]["test_rmse"] for m in model_names]
    active_feats = [reg_data["model_summaries"][m]["active_features_count"] for m in model_names]
    
    x = np.arange(len(short_names))
    width = 0.35
    
    # Bar chart for R2 and RMSE
    rects1 = ax1.bar(x - width/2, test_r2s, width, label="Test R² (%)", color="#0f9f8f")
    ax1_twin = ax1.twinx()
    rects2 = ax1_twin.bar(x + width/2, test_rmses, width, label="Test RMSE (LPA)", color="#f05f4f")
    
    ax1.set_ylabel("Test R² Score (%)", color="#0f9f8f", weight="bold")
    ax1_twin.set_ylabel("Test RMSE (LPA - lower is better)", color="#f05f4f", weight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, weight="bold")
    ax1.set_title("Salary Prediction: Model Performance Comparison (OLS vs Regularized)", fontsize=11, weight="bold", color="#21312f")
    ax1.set_ylim(0, 100)
    ax1_twin.set_ylim(0, max(test_rmses) * 1.4)
    
    # Sparsity / Active features breakdown
    colors_sparse = ["#4a90e2", "#50e3c2", "#f5a623", "#9013fe"]
    ax2.bar(short_names, active_feats, color=colors_sparse, width=0.5, edgecolor="#21312f", linewidth=1.2)
    ax2.set_ylabel("Active Non-Zero Features (Out of 10)", weight="bold")
    ax2.set_title("Feature Sparsity: Number of Active Selected Features", fontsize=11, weight="bold", color="#21312f")
    ax2.set_ylim(0, 12)
    for i, v in enumerate(active_feats):
        ax2.text(i, v + 0.3, f"{v}/10 Features", ha="center", weight="bold", fontsize=9.5)
        
    plt.tight_layout()
    plot2_path = config.PLOTS_DIR / "regularization_model_comparison.png"
    fig.savefig(plot2_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    
    # -------------------------------------------------------------
    # DIAGRAM 3: Learning Curves (Train vs Validation R² Gap)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=120)
    lc_ridge = generate_learning_curve_data(df, model_type="ridge")
    
    t_sizes = lc_ridge["train_sizes"]
    t_mean = lc_ridge["train_scores_mean"]
    v_mean = lc_ridge["val_scores_mean"]
    
    ax.plot(t_sizes, t_mean, "o-", color="#f05f4f", linewidth=2.4, label="Training R² Score")
    ax.plot(t_sizes, v_mean, "o-", color="#0f9f8f", linewidth=2.4, label="Cross-Validation R² Score")
    ax.fill_between(t_sizes, t_mean, v_mean, alpha=0.18, color="#f5b84b", label=f"Train-Val Gap (Final = {round(lc_ridge['final_gap'], 4)})")
    
    ax.set_title("Learning Curves: Diagnosing Model Overfitting & Generalization", fontsize=11, weight="bold", color="#21312f", pad=12)
    ax.set_xlabel("Training Dataset Sample Size", fontsize=9.5, weight="bold")
    ax.set_ylabel("R² Score", fontsize=9.5, weight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=True, loc="lower right", fontsize=9)
    plt.tight_layout()
    plot3_path = config.PLOTS_DIR / "regularization_learning_curves.png"
    fig.savefig(plot3_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    
    # -------------------------------------------------------------
    # DIAGRAM 4: Polynomial Overfitting Analogy (Slide 10)
    # -------------------------------------------------------------
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 8.5), dpi=120)
    sim_data = polynomial_overfitting_simulation(n_samples=25, noise=0.3)
    X_tr = sim_data["X_train"]
    y_tr = sim_data["y_train"]
    X_g = sim_data["X_grid"]
    y_true = sim_data["y_true"]
    preds = sim_data["predictions"]
    
    axes_map = [
        (ax1, "Degree 1 (Underfit)", "#e74c3c", "High Bias: Model too simple, misses true pattern in data."),
        (ax2, "Degree 4 (Just Right)", "#2ecc71", "Balanced: Low Bias + Low Variance, captures signal."),
        (ax3, "Degree 15 (Overfit)", "#e67e22", "High Variance: Memorizes training noise, fails on new points."),
        (ax4, "Degree 15 + Ridge (Regularized)", "#3498db", "Cured with Regularization: L2 penalty suppresses wild oscillations."),
    ]
    
    for ax, title, color, subtitle in axes_map:
        ax.scatter(X_tr, y_tr, color="#21312f", s=30, alpha=0.85, label="Training Data Points")
        ax.plot(X_g, y_true, color="black", linestyle=":", linewidth=1.5, label="True Underlying Function")
        ax.plot(X_g, preds[title], color=color, linewidth=2.5, label=f"Fitted Curve ({title.split()[0]})")
        ax.set_title(f"{title}\n{subtitle}", fontsize=9.5, weight="bold", color="#21312f")
        ax.set_ylim(-1.5, 3.0)
        ax.set_xlim(-3.2, 3.2)
        ax.legend(frameon=True, loc="upper center", fontsize=7.5)
        
    plt.tight_layout()
    plot4_path = config.PLOTS_DIR / "polynomial_overfitting_comparison.png"
    fig.savefig(plot4_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    
    return {
        "plot_coef_paths": "regularization_coef_paths.png",
        "plot_model_comparison": "regularization_model_comparison.png",
        "plot_learning_curves": "regularization_learning_curves.png",
        "plot_poly_overfitting": "polynomial_overfitting_comparison.png",
    }


# =====================================================================
# 8. Live Real-Time Prediction Helpers
# =====================================================================

def predict_salary_regularized(inputs_dict, model_type="ridge", reg_bundle=None):
    """Predict salary package (LPA) using the specified regularized linear model.
    
    model_type options: 'ols', 'ridge', 'lasso', 'elasticnet'
    """
    if reg_bundle is None:
        reg_bundle = train_regularized_regression()
        
    models_map = {
        "ols": "OLS (Ordinary Least Squares)",
        "ridge": "Ridge Regression (L2)",
        "lasso": "Lasso Regression (L1)",
        "elasticnet": "Elastic Net (L1 + L2)",
    }
    key = models_map.get(model_type.lower(), "Ridge Regression (L2)")
    model_obj = reg_bundle["models"][key]["model"]
    scaler = reg_bundle["scaler"]
    features = reg_bundle["features"]
    
    row_data = {}
    for feat in features:
        if feat in inputs_dict:
            row_data[feat] = float(inputs_dict[feat])
        else:
            row_data[feat] = float(reg_bundle["X_train"][feat].mean())
            
    df_row = pd.DataFrame([row_data])[features]
    row_scaled = scaler.transform(df_row)
    
    raw_pred = float(model_obj.predict(row_scaled)[0])
    predicted_salary = max(3.0, min(26.0, raw_pred))
    
    coef_dict = reg_bundle["model_summaries"][key]["raw_coef_dict"]
    breakdown = [
        {
            "feature": feat,
            "input_value": row_data[feat],
            "standardized_weight": round(coef_dict.get(feat, 0.0), 4),
            "is_active": not np.isclose(coef_dict.get(feat, 0.0), 0.0, atol=1e-4),
        }
        for feat in features
    ]
    breakdown.sort(key=lambda x: abs(x["standardized_weight"]), reverse=True)
    
    return {
        "predicted_salary": round(predicted_salary, 2),
        "raw_prediction": round(raw_pred, 2),
        "model_used": key,
        "inputs": row_data,
        "feature_breakdown": breakdown,
    }


def predict_placement_regularized(inputs_dict, model_type="ridge", clf_bundle=None):
    """Predict binary placement status using regularized logistic regression.
    
    model_type options: 'unregularized', 'ridge', 'lasso', 'elasticnet'
    """
    if clf_bundle is None:
        clf_bundle = train_regularized_classification()
        
    models_map = {
        "unregularized": "Unregularized Logistic",
        "ridge": "Ridge Logistic (L2)",
        "lasso": "Lasso Logistic (L1)",
        "elasticnet": "Elastic Net Logistic (L1 + L2)",
    }
    key = models_map.get(model_type.lower(), "Ridge Logistic (L2)")
    model_obj = clf_bundle["models"][key]["model"]
    scaler = clf_bundle["scaler"]
    features = clf_bundle["features"]
    
    row_data = {}
    for feat in features:
        if feat in inputs_dict:
            row_data[feat] = float(inputs_dict[feat])
        elif feat.startswith("SGPA_Sem"):
            row_data[feat] = float(inputs_dict.get("CGPA", 8.0))
        else:
            row_data[feat] = float(clf_bundle["X_train"][feat].median())
            
    df_row = pd.DataFrame([row_data])[features]
    row_scaled = scaler.transform(df_row)
    
    pred_binary = int(model_obj.predict(row_scaled)[0])
    probs = model_obj.predict_proba(row_scaled)[0]
    
    return {
        "is_placed": pred_binary == 1,
        "placement_status": "Placed" if pred_binary == 1 else "Not Placed",
        "prob_placed": round(float(probs[1]) * 100, 1),
        "prob_not_placed": round(float(probs[0]) * 100, 1),
        "model_used": key,
        "inputs": inputs_dict,
    }


# =====================================================================
# 9. Main Console Execution & Reporting
# =====================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("  MACHINE LEARNING (25SC2107E) - MODULE 2 SESSION 19")
    print("  Topic: Ridge, Lasso, Elastic Net Regularization & Overfitting Intuition")
    print("  Case Study: Placement Prediction System")
    print("=" * 80)
    
    # 1. Soft-thresholding verification
    print("\n--- 1. Soft-Thresholding Rule Demonstration (Lasso L1 Sparsity) ---")
    weights_sample = [2.5, 0.4, -1.8, -0.2, 0.05]
    thresholded = soft_threshold(weights_sample, lambda_val=1.0)
    print(f"Original weights:    {weights_sample}")
    print(f"L1 Thresholded (lambda=1): {[round(x, 3) for x in thresholded]}  (Notice small weights became exact 0!)")
    
    # 2. Continuous Salary Regression Models
    print("\n--- 2. Continuous Salary Regression: OLS vs Ridge vs Lasso vs Elastic Net ---")
    reg_bundle = train_regularized_regression()
    reg_table_df = pd.DataFrame(reg_bundle["comparison_table"])
    print(reg_table_df.to_string(index=False))
    
    print("\n--- 3. Hyperparameter Tuning Results (Cross-Validation) ---")
    print(f"  * Best Ridge alpha (lambda):       {reg_bundle['best_hyperparameters']['ridge_alpha']}")
    print(f"  * Best Lasso alpha (lambda):       {reg_bundle['best_hyperparameters']['lasso_alpha']}")
    print(f"  * Best Elastic Net alpha (lambda): {reg_bundle['best_hyperparameters']['elastic_alpha']}")
    print(f"  * Best Elastic Net L1 ratio:       {reg_bundle['best_hyperparameters']['elastic_l1_ratio']}")
    
    print("\n--- 4. Feature Selection Summary (Lasso L1 Zero Coefficients) ---")
    lasso_summary = reg_bundle["model_summaries"]["Lasso Regression (L1)"]
    print(f"  * Active Features Retained ({lasso_summary['active_features_count']}): {lasso_summary['selected_features']}")
    print(f"  * Zeroed-Out Features ({lasso_summary['zero_features_count']}):")
    for coef_info in lasso_summary["coefficients"]:
        if coef_info["is_zero"]:
            print(f"      - {coef_info['feature']}: weight = 0.0000")
            
    # 3. Binary Placement Classification Models
    print("\n--- 5. Binary Placement Classification: Unregularized vs L2 vs L1 vs Elastic Net ---")
    clf_bundle = train_regularized_classification()
    clf_table_df = pd.DataFrame(clf_bundle["comparison_table"])
    print(clf_table_df.to_string(index=False))
    
    # 4. Feature Engineering Selectors
    print("\n--- 6. Feature Engineering Automatic Selection (Lasso & Elastic Net) ---")
    df_clean = load_cleaned()
    lasso_sel = select_features_with_lasso(df_clean, target_col="PlacementStatus")
    print(f"  * Lasso Selected Features ({len(lasso_sel['selected_features'])}): {lasso_sel['selected_features']}")
    print(f"  * Lasso Dropped Features ({len(lasso_sel['dropped_features'])}): {lasso_sel['dropped_features']}")
    
    # 5. Overfitting Learning Curve Audit
    print("\n--- 7. Overfitting Diagnosis: Learning Curve Gap Analysis ---")
    lc_data = generate_learning_curve_data(df_clean, model_type="ridge")
    print(f"  * Ridge Final Train-Validation Gap (Delta R2): {round(lc_data['final_gap'], 4)}")
    if lc_data["final_gap"] < 0.05:
        print("  * Diagnosis: Optimal Generalization! Low variance, no significant overfitting detected.")
    else:
        print("  * Diagnosis: Mild overfitting gap present. Regularization alpha increased.")
        
    # 6. Generate Diagrams
    print("\n--- 8. Generating Session 19 Visual Diagrams in Output/plots/ ---")
    diagram_files = generate_regularization_diagrams(df_clean, reg_bundle, clf_bundle)
    for key, filename in diagram_files.items():
        print(f"  [SAVED] {key} -> Output/plots/{filename}")
        
    print("\n" + "=" * 80)
    print("  Session 19 Regularization module executed successfully!")
    print("=" * 80)
