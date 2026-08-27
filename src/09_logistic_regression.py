# Logistic Regression - sigmoid, cross-entropy, decision boundary, softmax
import sys, os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
import config
from src.data_utils import load_cleaned

LOGISTIC_DIR = config.BASE_DIR / "Output" / "logistic_regression"
LOGISTIC_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Load data and prepare features (numeric columns only) ----------

df = load_cleaned()
feature_cols = [
    "SGPA_Sem1", "SGPA_Sem2", "SGPA_Sem3", "SGPA_Sem4",
    "SGPA_Sem5", "SGPA_Sem6", "SGPA_Sem7", "SGPA_Sem8",
    "CGPA", "AttendancePercent", "Internships", "Projects",
    "Workshops", "Certifications", "Publications",
    "AptitudeTestScore", "SoftSkillsRating", "CodingTestScore",
    "MockInterviewScore", "ExtraCurricular",
]
available_features = [f for f in feature_cols if f in df.columns]

X = df[available_features].copy()
y = df["PlacementStatus"]

# Fill missing values with median
for col in available_features:
    X[col] = X[col].fillna(X[col].median())

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---------- Class balance: how many students are Placed vs Not Placed ----------

placed_pct = (y_train == 1).mean() * 100
print(f"Placed: {round(placed_pct,1)}%  |  Not Placed: {round(100-placed_pct,1)}%")

# ---------- CGPA vs Placement: the empirical S-curve ----------

bins = pd.cut(x_train["CGPA"], bins=15)
fraction_placed = y_train.groupby(bins, observed=True).mean()
bin_centers = [interval.mid for interval in fraction_placed.index]

plt.figure(figsize=(7, 5))
plt.plot(bin_centers, fraction_placed.values, marker="o", color="#0f9f8f")
plt.xlabel("CGPA")
plt.ylabel("Fraction Placed")
plt.title("CGPA vs Placement - the S-shaped pattern")
plt.savefig(LOGISTIC_DIR / "cgpa_vs_placement_curve.png")
plt.close()
print("CGPA vs placement S-curve saved")

# ---------- Which single feature predicts placement best? (AUC) ----------

auc_features = ["CGPA", "MockInterviewScore", "CodingTestScore", "AptitudeTestScore", "AttendancePercent"]
print("\nSingle-feature AUC (1.0 = perfect, 0.5 = no better than chance):")
auc_scores = {}
for col in auc_features:
    if col in x_train.columns:
        auc = roc_auc_score(y_train, x_train[col])
        auc_scores[col] = auc
        print(f"{col:<20}: {round(auc, 4)}")

# ---------- Why not a straight line? ----------

linreg = LinearRegression()
linreg.fit(x_train[["CGPA"]], y_train)
line_predictions = linreg.predict(x_train[["CGPA"]])
print(f"\nLinear regression on CGPA -> PlacementStatus gives predictions from "
      f"{round(line_predictions.min(),2)} to {round(line_predictions.max(),2)} "
      f"(a probability must stay between 0 and 1 - this is why we need logistic regression)")

# ---------- The sigmoid function ----------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z_values = np.linspace(-8, 8, 100)
plt.figure(figsize=(7, 5))
plt.plot(z_values, sigmoid(z_values), color="#f05f4f", linewidth=2.5)
plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
plt.axvline(0, color="gray", linestyle="--", linewidth=1)
plt.xlabel("z")
plt.ylabel("sigmoid(z)")
plt.title("The Sigmoid Function - always between 0 and 1")
plt.savefig(LOGISTIC_DIR / "sigmoid_function.png")
plt.close()
print("Sigmoid function plot saved")

# ---------- Fit the full logistic regression model ----------

model = LogisticRegression(max_iter=1000, random_state=42)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)

model.fit(x_train_scaled, y_train)

val_accuracy = accuracy_score(y_val, model.predict(x_val_scaled))
val_log_loss = log_loss(y_val, model.predict_proba(x_val_scaled))
print(f"\nFull model - Validation accuracy: {round(val_accuracy, 4)}")
print(f"Full model - Cross-entropy (log loss): {round(val_log_loss, 4)}")

# ---------- Decision boundary using 2 features ----------

boundary_model = LogisticRegression(max_iter=1000, random_state=42)
boundary_model.fit(x_train[["CGPA", "CodingTestScore"]].to_numpy(), y_train)

x_min, x_max = x_train["CGPA"].min() - 0.5, x_train["CGPA"].max() + 0.5
y_min, y_max = x_train["CodingTestScore"].min() - 5, x_train["CodingTestScore"].max() + 5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid_predictions = boundary_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(7, 6))
plt.contourf(xx, yy, grid_predictions, alpha=0.2, levels=1, colors=["#f4a6a6", "#a6c8f4"])
plt.scatter(x_val["CGPA"], x_val["CodingTestScore"], c=y_val, cmap="coolwarm", alpha=0.3, s=10)
plt.xlabel("CGPA")
plt.ylabel("Coding Test Score")
plt.title("Decision Boundary (CGPA + Coding Test Score)")
plt.savefig(LOGISTIC_DIR / "decision_boundary.png")
plt.close()
print("Decision boundary plot saved")

# ---------- Scaler comparison: Unscaled vs StandardScaler vs MinMaxScaler ----------

def evaluate_logistic_regression(x_tr, x_va, y_tr, y_va, label):
    m = LogisticRegression(max_iter=1000, random_state=42)
    m.fit(x_tr, y_tr)
    acc = accuracy_score(y_va, m.predict(x_va))
    print(f"{label:<20}: Validation Accuracy = {round(acc, 4)}")
    return acc

print("\nScaler comparison (same model, three different inputs):")
results = {}
results["Unscaled"] = evaluate_logistic_regression(x_train, x_val, y_train, y_val, "Unscaled")

standard_scaler = StandardScaler()
x_train_std = standard_scaler.fit_transform(x_train)
x_val_std = standard_scaler.transform(x_val)
results["StandardScaler"] = evaluate_logistic_regression(x_train_std, x_val_std, y_train, y_val, "StandardScaler")

minmax_scaler = MinMaxScaler()
x_train_mm = minmax_scaler.fit_transform(x_train)
x_val_mm = minmax_scaler.transform(x_val)
results["MinMaxScaler"] = evaluate_logistic_regression(x_train_mm, x_val_mm, y_train, y_val, "MinMaxScaler")

# ---------- Multinomial logistic regression (softmax), 3 classes ----------

salary_median = df.loc[df["Salary Package"] > 0, "Salary Package"].median() if "Salary Package" in df.columns else 8.5

def make_package_tier(row):
    if row.get("PlacementStatus", 0) == 0 or row.get("Salary Package", 0) == 0:
        return "Not Placed"
    elif row.get("Salary Package", 0) < salary_median:
        return "Standard Package"
    return "Premium Package"

y_tier = df.apply(make_package_tier, axis=1)
y_train_tier = y_tier.loc[x_train.index]
y_val_tier = y_tier.loc[x_val.index]

softmax_model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
softmax_model.fit(x_train_std, y_train_tier)
tier_accuracy = accuracy_score(y_val_tier, softmax_model.predict(x_val_std))
print(f"\nSoftmax (3-class: Not Placed / Standard / Premium) accuracy: {round(tier_accuracy, 4)}")

# ---------- Save a report with all key numbers ----------

with open(LOGISTIC_DIR / "logistic_regression_report.txt", "w") as f:
    f.write(f"Placed: {round(placed_pct,1)}%  Not Placed: {round(100-placed_pct,1)}%\n\n")
    f.write("Single-feature AUC:\n")
    for col, auc in auc_scores.items():
        f.write(f"  {col}: {round(auc,4)}\n")
    f.write(f"\nFull model validation accuracy: {round(val_accuracy,4)}\n")
    f.write(f"Full model cross-entropy (log loss): {round(val_log_loss,4)}\n")
    f.write("\nScaler comparison:\n")
    for label, acc in results.items():
        f.write(f"  {label}: {round(acc,4)}\n")
    f.write(f"\nSoftmax 3-class accuracy: {round(tier_accuracy,4)}\n")
