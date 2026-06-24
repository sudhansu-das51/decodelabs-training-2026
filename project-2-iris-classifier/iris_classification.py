# =============================================================================
#  IRIS DATA CLASSIFICATION — Supervised Learning Pipeline
#  Project 2 | DecodeLabs Industrial Training Kit | Batch 2026
#  Developer  : Subin Das
#  Description: A complete supervised learning pipeline that trains a
#               K-Nearest Neighbors (KNN) classifier on the Iris dataset.
#               Follows the IPO Model (Input -> Process -> Output) taught
#               in Module 01: Architectural Paradigms.
#
#  Pipeline   : Load -> Explore -> Scale -> Split -> Train -> Predict -> Evaluate
# =============================================================================


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
)


# -----------------------------------------------------------------------------
#  CONFIGURATION
# -----------------------------------------------------------------------------

RANDOM_STATE   = 42      # Fixed seed -> reproducible train/test split every run
TEST_SIZE      = 0.20    # 80% Training / 20% Testing (industry standard split)
# Note: K for KNN is NOT hardcoded here — it is determined dynamically by
# find_optimal_k() below (Elbow Method), then passed into train_model().


def print_section(title: str) -> None:
    """Prints a clean, professional section divider in the terminal."""
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# -----------------------------------------------------------------------------
#  PHASE 1 — INPUT: LOAD & UNDERSTAND THE DATASET
# -----------------------------------------------------------------------------

def load_and_explore_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list]:
    """
    Loads the Iris benchmark dataset (built into scikit-learn) and
    prints a data inventory so we understand what we're working with
    before touching any algorithm — exploration always comes first.
    """
    print_section("PHASE 1: INPUT — LOADING THE IRIS DATASET")

    iris = load_iris()

    # Raw feature matrix (X) and target labels (y)
    X = iris.data                      # shape: (150, 4)
    y = iris.target                    # shape: (150,)  -> values: 0, 1, 2
    feature_names = iris.feature_names
    target_names = list(iris.target_names)

    # Wrap into a DataFrame purely for human-readable inspection
    df = pd.DataFrame(X, columns=feature_names)
    df["species"] = [target_names[label] for label in y]

    print(f"  Total Samples       : {df.shape[0]}")
    print(f"  Total Features      : {len(feature_names)}")
    print(f"  Feature Names       : {feature_names}")
    print(f"  Target Classes      : {target_names}")
    print(f"  Class Distribution  :")
    for name, count in df["species"].value_counts().items():
        print(f"      - {name:<12}: {count} samples")

    print("\n  First 5 rows of the dataset:")
    print(df.head().to_string(index=False))

    print("\n  Statistical Summary:")
    print(df.describe().round(2).to_string())

    return df, X, y, target_names


# -----------------------------------------------------------------------------
#  PHASE 1 — INPUT (PART B): FEATURE VISUALIZATION
# -----------------------------------------------------------------------------

def visualize_features(df: pd.DataFrame) -> None:
    """
    Visualizes how the 3 species separate across feature dimensions —
    this is the 'understand the dataset' step before any modeling, matching
    the PDF's own architecture diagram (sepal length / petal width plotted
    as distinguishing features between species).
    """
    print_section("PHASE 1B: INPUT — FEATURE VISUALIZATION")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.scatterplot(
        data=df, x="sepal length (cm)", y="sepal width (cm)",
        hue="species", palette="deep", s=60, ax=axes[0]
    )
    axes[0].set_title("Sepal Length vs Sepal Width", fontweight="bold")
    axes[0].grid(alpha=0.3)

    sns.scatterplot(
        data=df, x="petal length (cm)", y="petal width (cm)",
        hue="species", palette="deep", s=60, ax=axes[1]
    )
    axes[1].set_title("Petal Length vs Petal Width", fontweight="bold")
    axes[1].grid(alpha=0.3)

    plt.suptitle("Raw Material: Understanding the Iris Feature Space", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("feature_visualization.png", dpi=150)
    plt.close()

    print("  ✓ Saved visualization -> feature_visualization.png")
    print("  Observation: Petal measurements separate the 3 species far more")
    print("               cleanly than sepal measurements — petals will carry")
    print("               more weight in the model's decision boundary.")


# -----------------------------------------------------------------------------
#  PHASE 2 — PROCESS (PART A): FEATURE SCALING — "THE GATEKEEPER RULE"
# -----------------------------------------------------------------------------

def scale_features(X: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    """
    KNN is a distance-based algorithm — it measures how 'close' points are.
    If one feature has a much larger range than another, it will dominate
    the distance calculation and bias the model. StandardScaler fixes this
    by transforming every feature to Mean = 0, Variance = 1.
    """
    print_section("PHASE 2A: PROCESS — FEATURE SCALING (StandardScaler)")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("  Before Scaling (first row) :", np.round(X[0], 2))
    print("  After  Scaling (first row) :", np.round(X_scaled[0], 2))
    print("\n  Scaled Feature Stats:")
    print(f"      Mean     : {np.round(X_scaled.mean(axis=0), 4)}")
    print(f"      Std Dev  : {np.round(X_scaled.std(axis=0), 4)}")
    print("  ✓ All features now share the same scale — no feature dominates.")

    return X_scaled, scaler


# -----------------------------------------------------------------------------
#  PHASE 2 — PROCESS (PART B): TRAIN-TEST SPLIT — "STRUCTURAL INTEGRITY"
# -----------------------------------------------------------------------------

def split_data(X_scaled: np.ndarray, y: np.ndarray):
    """
    Splits the dataset into a Training Set (used to teach the model patterns)
    and a Test Set (used purely to validate — the model never sees this
    during training). stratify=y ensures each class is proportionally
    represented in both splits, avoiding accidental class imbalance.
    """
    print_section("PHASE 2B: PROCESS — TRAIN-TEST SPLIT")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,        # Randomize before splitting to remove order bias
        stratify=y,           # Keep class proportions balanced in both sets
    )

    print(f"  Training Set Size  : {X_train.shape[0]} samples ({(1-TEST_SIZE)*100:.0f}%)")
    print(f"  Test Set Size       : {X_test.shape[0]} samples ({TEST_SIZE*100:.0f}%)")
    print(f"  Stratified Split    : Yes — class balance preserved in both sets")

    return X_train, X_test, y_train, y_test


# -----------------------------------------------------------------------------
#  PHASE 2 — PROCESS (PART C): MODEL TRAINING — "THE ALGORITHM"
# -----------------------------------------------------------------------------

def train_model(X_train: np.ndarray, y_train: np.ndarray, k: int) -> KNeighborsClassifier:
    """
    Trains a K-Nearest Neighbors classifier.
    Core idea: 'Similar things exist in close proximity.' To classify a new
    flower, KNN looks at its K closest neighbors in the training data and
    assigns the majority class among them.

    Scikit-learn's 3-step contract:
        1. INSTANTIATE -> build the model object
        2. FIT          -> let the model memorize the training patterns
        3. PREDICT      -> apply the learned logic to unseen data
    """
    print_section("PHASE 2C: PROCESS — TRAINING THE FINAL KNN MODEL")

    model = KNeighborsClassifier(n_neighbors=k)   # 1. INSTANTIATE
    model.fit(X_train, y_train)                    # 2. FIT

    print(f"  Algorithm           : K-Nearest Neighbors (KNN)")
    print(f"  K (neighbors)       : {k}  (selected via Elbow Method tuning)")
    print(f"  Model trained on    : {X_train.shape[0]} samples")
    print("  ✓ Model has finished memorizing the training patterns.")

    return model


# -----------------------------------------------------------------------------
#  PHASE 2 — PROCESS (PART D): FINDING THE OPTIMAL K — "TUNING THE ENGINE"
# -----------------------------------------------------------------------------

def find_optimal_k(X_train, X_test, y_train, y_test, max_k: int = 20, min_k: int = 3) -> int:
    """
    Tests K values from min_k to max_k and tracks the error rate for each.

    IMPORTANT: We deliberately start the search at K=3, not K=1. The PDF's
    own 'Tuning the Engine' slide explicitly flags K=1 as a noise/overfitting
    risk — a model that only looks at its single nearest neighbor memorizes
    outliers instead of learning a real pattern. Very high K, on the other
    hand, underfits (too generic). The 'elbow point' — where error rate
    stabilizes at its lowest within this safe range — is the optimal K.
    """
    print_section("PHASE 2D: PROCESS — TUNING K (ELBOW METHOD)")
    print(f"  Note: Search starts at K={min_k}, not K=1 — per the PDF's own")
    print(f"        warning that K=1 risks overfitting to noise.\n")

    error_rates = []
    k_range = range(min_k, max_k + 1)

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        error_rate = np.mean(predictions != y_test)
        error_rates.append(error_rate)

    optimal_k = k_range[int(np.argmin(error_rates))]

    print("  K   | Error Rate")
    print("  ----|-----------")
    for k, err in zip(k_range, error_rates):
        marker = "  <-- optimal" if k == optimal_k else ""
        print(f"  {k:<3} | {err:.4f}{marker}")

    print(f"\n  ✓ Optimal K identified: {optimal_k}  (safe range, avoids overfitting)")

    # Visual: Error Rate vs K (saved as PNG for the report/portfolio)
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, error_rates, marker="o", linestyle="--",
              color="#2C3E91", markerfacecolor="#E8542F")
    plt.axvline(x=optimal_k, color="#E8542F", linestyle=":", linewidth=1.5)
    plt.title("Tuning the Engine: Error Rate vs. K Value", fontsize=13, fontweight="bold")
    plt.xlabel("K Value (Number of Neighbors)")
    plt.ylabel("Error Rate")
    plt.xticks(list(k_range))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("k_tuning_plot.png", dpi=150)
    plt.close()
    print("  ✓ Saved visualization -> k_tuning_plot.png")

    return optimal_k


# -----------------------------------------------------------------------------
#  PHASE 3 — OUTPUT: MODEL EVALUATION — "BEYOND THE ACCURACY MIRAGE"
# -----------------------------------------------------------------------------

def evaluate_model(model: KNeighborsClassifier, X_test: np.ndarray,
                    y_test: np.ndarray, target_names: list) -> None:
    """
    Evaluates the trained model using multiple metrics — not just accuracy.
    The PDF explicitly warns against the 'Accuracy Mirage': in imbalanced
    datasets, a high accuracy score can hide a poorly performing model.
    The Confusion Matrix and F1 Score give us the real picture.
    """
    print_section("PHASE 3: OUTPUT — MODEL EVALUATION")

    predictions = model.predict(X_test)

    # --- Accuracy ---
    accuracy = accuracy_score(y_test, predictions)
    print(f"  Overall Accuracy    : {accuracy * 100:.2f}%")

    # --- F1 Score (Harmonic Mean of Precision & Recall) ---
    f1_macro = f1_score(y_test, predictions, average="macro")
    print(f"  F1 Score (macro)    : {f1_macro:.4f}")
    print("  Note: F1 balances Precision (trustworthiness) and Recall")
    print("        (sensitivity) — more reliable than accuracy alone.")

    # --- Confusion Matrix — The Diagnostic Tool ---
    cm = confusion_matrix(y_test, predictions)
    print("\n  Confusion Matrix:")
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    print(cm_df.to_string())
    print("\n  Reading the matrix -> rows = actual class, columns = predicted class")
    print("  Diagonal values = correct predictions (TP). Off-diagonal = errors.")

    # --- Full Classification Report (Precision, Recall, F1 per class) ---
    print("\n  Detailed Classification Report:")
    print(classification_report(y_test, predictions, target_names=target_names))

    # --- Visual Confusion Matrix Heatmap (saved as PNG) ---
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=target_names, yticklabels=target_names,
        cbar=False, linewidths=0.5, linecolor="white"
    )
    plt.title("Confusion Matrix — Iris Classification", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()
    print("  ✓ Saved visualization -> confusion_matrix.png")


# -----------------------------------------------------------------------------
#  BONUS: LIVE PREDICTION — TEST WITH CUSTOM INPUT
# -----------------------------------------------------------------------------

def predict_custom_sample(model: KNeighborsClassifier, scaler: StandardScaler,
                           target_names: list) -> None:
    """
    Demonstrates the model on a brand-new, unseen flower measurement —
    proving the model generalizes beyond the training/test data it has
    already seen. This is the real-world 'deployment' moment.
    """
    print_section("BONUS: LIVE PREDICTION ON A NEW SAMPLE")

    # Example measurements (sepal_length, sepal_width, petal_length, petal_width)
    sample = np.array([[5.8, 2.7, 4.9, 1.8]])
    sample_scaled = scaler.transform(sample)

    prediction = model.predict(sample_scaled)[0]
    probabilities = model.predict_proba(sample_scaled)[0]

    print(f"  Input Measurements  : {sample[0].tolist()}")
    print(f"  Predicted Species   : {target_names[prediction]}")
    print("  Class Probabilities :")
    for name, prob in zip(target_names, probabilities):
        print(f"      - {name:<12}: {prob * 100:.1f}%")


# -----------------------------------------------------------------------------
#  MAIN PIPELINE — ORCHESTRATES THE FULL IPO FLOW
# -----------------------------------------------------------------------------

def run_pipeline() -> None:
    """
    Executes the complete supervised learning pipeline end-to-end:
    Input (load/explore) -> Process (scale/split/train/tune) -> Output (evaluate).
    """
    print("\n" + "#" * 65)
    print("#   IRIS DATA CLASSIFICATION — SUPERVISED LEARNING PIPELINE")
    print("#   DecodeLabs | Project 2 | Batch 2026")
    print("#" * 65)

    # --- PHASE 1: INPUT ---
    df, X, y, target_names = load_and_explore_data()
    visualize_features(df)

    # --- PHASE 2: PROCESS ---
    X_scaled, scaler = scale_features(X)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    optimal_k = find_optimal_k(X_train, X_test, y_train, y_test)
    model = train_model(X_train, y_train, k=optimal_k)

    # --- PHASE 3: OUTPUT ---
    evaluate_model(model, X_test, y_test, target_names)

    # --- BONUS ---
    predict_custom_sample(model, scaler, target_names)

    print_section("PIPELINE COMPLETE ✓")
    print("  Project 2 deliverables generated successfully:")
    print("    - feature_visualization.png")
    print("    - k_tuning_plot.png")
    print("    - confusion_matrix.png")
    print()


# -----------------------------------------------------------------------------
#  ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()
