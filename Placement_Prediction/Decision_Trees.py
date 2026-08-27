"""Decision Trees - Placement Prediction (Project Dataset)

Concepts from Module 3 Sessions 25-26:
- Splitting criteria: Entropy, Information Gain, Gini Index, and MSE
- Tree depth, pruning, and overfitting diagnostics
- Feature importance analysis on Placement Prediction dataset
"""

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import load_cleaned

# ---------- 1. Load Project Placement Dataset ----------
print("Loading project placement dataset...")
df = load_cleaned()
print(f"Dataset shape: {df.shape}")

# Define feature columns and target
feature_cols = [
    "CGPA", "CodingTestScore", "MockInterviewScore", "AptitudeTestScore",
    "AttendancePercent", "SoftSkillsRating", "Internships", "Projects"
]
target_col = "PlacementStatus"

# Impute and prepare features
for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

X = df[feature_cols]
y = df[target_col].astype(int)

# Split into Training and Validation sets
x_train, x_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Features: {feature_cols}")
print(f"Train samples: {len(x_train)}, Validation samples: {len(x_val)}")
print("\n" + "=" * 60 + "\n")

# ---------- 2. Unconstrained Tree (Overfitting Demonstration) ----------
full_tree = DecisionTreeClassifier(criterion="entropy", random_state=42)
full_tree.fit(x_train, y_train)

train_acc_full = accuracy_score(y_train, full_tree.predict(x_train))
val_acc_full = accuracy_score(y_val, full_tree.predict(x_val))

print("Unconstrained Tree (No Depth Limit):")
print(f"  Tree Depth:     {full_tree.get_depth()}")
print(f"  Leaves Count:   {full_tree.get_n_leaves()}")
print(f"  Train Accuracy: {round(train_acc_full * 100, 2)}%")
print(f"  Val Accuracy:   {round(val_acc_full * 100, 2)}%")
print(f"  Overfitting Gap: {round((train_acc_full - val_acc_full) * 100, 2)}%")
print("\n" + "=" * 60 + "\n")

# ---------- 3. Regularized Shallow Tree (max_depth=3) ----------
shallow_tree = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
shallow_tree.fit(x_train, y_train)

train_acc_shallow = accuracy_score(y_train, shallow_tree.predict(x_train))
val_acc_shallow = accuracy_score(y_val, shallow_tree.predict(x_val))
val_preds = shallow_tree.predict(x_val)

print("Regularized Tree (max_depth=3, criterion='entropy'):")
print(f"  Train Accuracy: {round(train_acc_shallow * 100, 2)}%")
print(f"  Val Accuracy:   {round(val_acc_shallow * 100, 2)}%")
print(f"  Precision:      {round(precision_score(y_val, val_preds) * 100, 2)}%")
print(f"  Recall:         {round(recall_score(y_val, val_preds) * 100, 2)}%")
print(f"  F1 Score:       {round(f1_score(y_val, val_preds) * 100, 2)}%")
print("\nDecision Tree Rules:\n" + export_text(shallow_tree, feature_names=feature_cols))
print("=" * 60 + "\n")

# ---------- 4. Feature Importance ----------
importance = pd.Series(shallow_tree.feature_importances_, index=feature_cols)
top_features = importance[importance > 0].sort_values(ascending=False)

print("Top Features by Impurity Reduction:")
for feature, score in top_features.items():
    print(f"  {feature:<22} {score:.4f} ({score*100:.1f}%)")

# ---------- 5. Visualizations ----------
# Tree Plot
plt.figure(figsize=(14, 7))
plot_tree(
    shallow_tree,
    feature_names=feature_cols,
    class_names=["Not Placed", "Placed"],
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title("Placement Prediction Decision Tree (max_depth=3, Entropy)")
plt.tight_layout()
plt.show()

# Feature Importance Plot
plt.figure(figsize=(8, 4.5))
top_features.sort_values().plot(kind="barh", color="#0f766e")
plt.xlabel("Importance Score (Entropy Gain)")
plt.title("Decision Tree Feature Importance (Placement Prediction)")
plt.tight_layout()
plt.show()
