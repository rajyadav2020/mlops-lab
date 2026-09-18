# ============================================
# LAB 3 - CLASSIFICATION
# Breast Cancer Dataset
# ============================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ============================================
# 1. LOAD DATASET
# ============================================

df = pd.read_csv("data/breast_cancer.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())


# ============================================
# 2. CLEAN DATA
# ============================================

# Remove unnecessary columns if present
for col in ["id", "ID", "Unnamed: 32"]:
    if col in df.columns:
        df = df.drop(columns=[col])


# ============================================
# 3. FEATURES AND TARGET
# ============================================

target = "diagnosis"

X = df.drop(columns=[target])
y = df[target]


# Convert:
# B = Benign = 0
# M = Malignant = 1

if y.dtype == "object":
    y = y.map({
        "B": 0,
        "M": 1
    })


print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)


# ============================================
# 4. TRAIN TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================
# 5. CREATE MODELS
# ============================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


# ============================================
# 6. TRAIN MODELS AND COMPARE ACCURACY
# ============================================

results = []

for name, model in models.items():

    # Train
    model.fit(X_train, y_train)

    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Accuracy
    train_accuracy = accuracy_score(
        y_train,
        train_pred
    )

    test_accuracy = accuracy_score(
        y_test,
        test_pred
    )

    results.append({
        "Model": name,
        "Train Accuracy": train_accuracy,
        "Test Accuracy": test_accuracy
    })

    print("\n==============================")
    print(name)
    print("==============================")

    print(
        "Train Accuracy:",
        round(train_accuracy, 4)
    )

    print(
        "Test Accuracy:",
        round(test_accuracy, 4)
    )


# ============================================
# 7. MODEL COMPARISON TABLE
# ============================================

results_df = pd.DataFrame(results)

print("\n\n===== MODEL COMPARISON =====")

print(
    results_df.round(4).to_string(index=False)
)


# ============================================
# 8. CROSS VALIDATION FOR TREE DEPTH
# ============================================

depths = [2, 3, 4, 5, 6, 8, 10, None]

depth_results = {}

for depth in depths:

    tree = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    scores = cross_val_score(
        tree,
        X,
        y,
        cv=5,
        scoring="accuracy"
    )

    depth_results[depth] = scores.mean()


print("\n\n===== TREE DEPTH CROSS-VALIDATION =====")

for depth, score in depth_results.items():

    print(
        "Depth:",
        depth,
        "| CV Accuracy:",
        round(score, 4)
    )


# ============================================
# 9. FIND BEST TREE DEPTH
# ============================================

best_depth = max(
    depth_results,
    key=depth_results.get
)

best_score = depth_results[best_depth]

print("\nBest Tree Depth:", best_depth)

print(
    "Best CV Accuracy:",
    round(best_score, 4)
)


# ============================================
# 10. TRAIN BEST DECISION TREE
# ============================================

best_tree = DecisionTreeClassifier(
    max_depth=best_depth,
    random_state=42
)

best_tree.fit(
    X_train,
    y_train
)

tree_pred = best_tree.predict(X_test)


# ============================================
# 11. CONFUSION MATRIX
# ============================================

cm = confusion_matrix(
    y_test,
    tree_pred
)

print("\n\n===== CONFUSION MATRIX =====")

print(cm)


# ============================================
# 12. CLASSIFICATION REPORT
# ============================================

print("\n\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        tree_pred,
        target_names=[
            "Benign",
            "Malignant"
        ]
    )
)