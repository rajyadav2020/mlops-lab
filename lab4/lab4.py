#classification of the data.py
import pandas as pd


def load_data(path="data/breast_cancer.csv"):
    """
    Load the Breast Cancer dataset.
    """
    df = pd.read_csv(path)

    return df

def get_features_and_target(df):
    """
    Separate features and target from the Breast Cancer dataset.
    """

    # Remove unnecessary columns if they exist
    for col in ["id", "ID", "Unnamed: 32"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Target column
    target = "diagnosis"

    # Features
    X = df.drop(columns=[target])

    # Target
    y = df[target]

    # Convert:
    # B = Benign = 0
    # M = Malignant = 1
    if y.dtype == "object":
        y = y.map({
            "B": 0,
            "M": 1
        })

    return X, y

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


def split_data(X, y):
    """
    Split dataset into training and testing sets.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def create_models():
    """
    Create the three classification models.
    """

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

    return models


def train_and_compare(X_train, X_test, y_train, y_test):
    """
    Train all classification models and compare
    training and testing accuracy.
    """

    models = create_models()

    results = []

    for name, model in models.items():

        # Train model
        model.fit(X_train, y_train)

        # Training prediction
        train_pred = model.predict(X_train)

        # Testing prediction
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

    return models, results


def find_best_tree_depth(X, y):
    """
    Find the best Decision Tree depth using
    5-fold cross-validation.
    """

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

    best_depth = max(
        depth_results,
        key=depth_results.get
    )

    return best_depth, depth_results


def train_best_tree(
    X_train,
    y_train,
    best_depth
):
    """
    Train Decision Tree using the best depth.
    """

    model = DecisionTreeClassifier(
        max_depth=best_depth,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def evaluate_model(model, X_test, y_test):
    """
    Generate confusion matrix and classification report.
    """

    predictions = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "Benign",
            "Malignant"
        ]
    )

    return cm, report

#validataion

def validate_data(X, y):
    """
    Validate the classification dataset.
    """

    if X.empty:
        raise ValueError("Feature dataset is empty.")

    if y.empty:
        raise ValueError("Target dataset is empty.")

    if X.isnull().sum().sum() > 0:
        raise ValueError("Features contain missing values.")

    if y.isnull().sum() > 0:
        raise ValueError("Target contains missing values.")

    unique_classes = set(y.unique())

    if not unique_classes.issubset({0, 1}):
        raise ValueError(
            "Target must contain only 0 and 1."
        )

    return True


#classification of initpy
from .data import load_data

from .features import (
    get_features_and_target
)

from .model import (
    split_data,
    create_models,
    train_and_compare,
    find_best_tree_depth,
    train_best_tree,
    evaluate_model
)

from .validate import (
    validate_data
)


#final breast cancer train
import pandas as pd

from classification import (
    load_data,
    get_features_and_target,
    validate_data,
    split_data,
    train_and_compare,
    find_best_tree_depth,
    train_best_tree,
    evaluate_model
)


# ==========================================
# 1. LOAD DATA
# ==========================================

df = load_data()

print("Dataset loaded successfully.")

print("Shape:", df.shape)


# ==========================================
# 2. GET FEATURES AND TARGET
# ==========================================

X, y = get_features_and_target(df)

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)


# ==========================================
# 3. VALIDATE DATA
# ==========================================

validate_data(X, y)

print("\nData validation successful.")


# ==========================================
# 4. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = split_data(
    X,
    y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 5. TRAIN AND COMPARE MODELS
# ==========================================

models, results = train_and_compare(
    X_train,
    X_test,
    y_train,
    y_test
)


# ==========================================
# 6. DISPLAY RESULTS
# ==========================================

results_df = pd.DataFrame(results)

print("\n===== MODEL COMPARISON =====")

print(
    results_df.round(4).to_string(index=False)
)


# ==========================================
# 7. FIND BEST TREE DEPTH
# ==========================================

best_depth, depth_results = find_best_tree_depth(
    X,
    y
)

print("\n===== TREE DEPTH CROSS-VALIDATION =====")

for depth, score in depth_results.items():

    print(
        "Depth:",
        depth,
        "| CV Accuracy:",
        round(score, 4)
    )

print("\nBest Tree Depth:", best_depth)

print(
    "Best CV Accuracy:",
    round(depth_results[best_depth], 4)
)


# ==========================================
# 8. TRAIN BEST TREE
# ==========================================

best_tree = train_best_tree(
    X_train,
    y_train,
    best_depth
)


# ==========================================
# 9. EVALUATE BEST TREE
# ==========================================

cm, report = evaluate_model(
    best_tree,
    X_test,
    y_test
)


# ==========================================
# 10. CONFUSION MATRIX
# ==========================================

print("\n===== CONFUSION MATRIX =====")

print(cm)


# ==========================================
# 11. CLASSIFICATION REPORT
# ==========================================

print("\n===== CLASSIFICATION REPORT =====")

print(report)

print("\nTraining and evaluation completed.")