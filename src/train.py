"""
train.py
========
Model training pipeline for the Sales Analytics Engine.

This module implements the complete training workflow:
  1. Load and preprocess data
  2. Build feature engineering pipeline
  3. Train a Logistic Regression classifier
  4. Evaluate on the test set
  5. Save the trained model

Baseline Model: Logistic Regression
------------------------------------

### Why Logistic Regression?

For a Phase 1 baseline, Logistic Regression is the ideal choice because:

1. **Interpretability**: The model learns a weight for each feature. Positive
   weights indicate features that push toward class 1 (WON), negative weights
   push toward class 0 (LOST). This is invaluable for stakeholder communication
   in business contexts — we can explain WHY a deal is predicted to be won/lost.

2. **Probabilistic output**: Unlike hard classifiers, LR outputs calibrated
   probabilities P(WON|features), enabling threshold tuning for different
   business objectives (e.g., high-recall for identifying at-risk deals).

3. **Efficiency**: Training is fast even on large feature spaces (500+ TF-IDF
   features). This allows rapid experimentation and iteration.

4. **Strong baseline**: Despite its simplicity, LR is surprisingly competitive
   on text classification tasks, especially with TF-IDF features. It establishes
   a meaningful performance floor for comparison with more complex models.

### Bias-Variance Tradeoff

Logistic Regression is a **low-variance, potentially-high-bias** model:

  - **Low variance**: The linear decision boundary is stable across different
    training samples. Small perturbations in data don't cause wild changes
    in predictions (unlike deep neural networks or decision trees).

  - **Potential high bias**: If the true decision boundary is highly nonlinear,
    LR will underfit. However, for text classification with TF-IDF features,
    linear boundaries are often sufficient because the high-dimensional
    feature space (500+ dimensions) provides enough expressivity.

  - **Regularization** (default L2 in sklearn) further controls variance by
    penalizing large weights, preventing overfitting to noise.

For Phase 1, we accept this tradeoff: slightly higher bias in exchange for
interpretability and robustness. Phase 2 can explore nonlinear models
(Random Forest, XGBoost, or neural networks) if the linear baseline is
insufficient.
"""

import os
import sys
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Add project root to path for module imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_data, inspect_data, handle_missing_values, split_data
from src.feature_engineering import build_feature_transformer
from src.evaluate import evaluate_model, print_evaluation_report, plot_confusion_matrix, display_win_probabilities

# ─── Configuration ──────────────────────────────────────────────────────────
MODEL_MAX_ITER = 200      # Maximum iterations for LR convergence
MODEL_RANDOM_STATE = 42   # Reproducibility
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "baseline_model.pkl")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def build_pipeline() -> Pipeline:
    """
    Build the complete ML pipeline: feature engineering → classifier.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A pipeline with two steps:
        1. 'features': ColumnTransformer (TF-IDF + Scaler + OneHot)
        2. 'classifier': Logistic Regression

    Notes
    -----
    Using a Pipeline ensures that:
    - Feature transformations are fitted ONLY on training data (no leakage)
    - The same transformations are applied consistently at prediction time
    - The entire model can be serialized/deserialized as a single object
    """
    # Build the feature transformer
    feature_transformer = build_feature_transformer()

    # Build the classifier
    classifier = LogisticRegression(
        max_iter=MODEL_MAX_ITER,
        random_state=MODEL_RANDOM_STATE,
        solver="lbfgs",       # Efficient for L2 regularization
        penalty="l2",         # Ridge regularization to prevent overfitting
        C=1.0,                # Regularization strength (inverse)
        class_weight=None,    # No class reweighting (Phase 1 baseline)
        verbose=0,
    )

    # Combine into pipeline
    pipeline = Pipeline([
        ("features", feature_transformer),
        ("classifier", classifier),
    ])

    print("\n✅ ML Pipeline built:")
    print("   Step 1: ColumnTransformer (TF-IDF + Scaler + OneHot)")
    print(f"   Step 2: LogisticRegression (max_iter={MODEL_MAX_ITER}, "
          f"penalty=L2, C=1.0)")

    return pipeline


def train(pipeline: Pipeline, X_train, y_train) -> Pipeline:
    """
    Train the ML pipeline on the training data.

    Parameters
    ----------
    pipeline : Pipeline
        The untrained sklearn Pipeline.
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training labels.

    Returns
    -------
    Pipeline
        The fitted pipeline.
    """
    print("\n⏳ Training model...")
    pipeline.fit(X_train, y_train)

    # Report training accuracy
    train_acc = pipeline.score(X_train, y_train)
    print(f"✅ Training complete!")
    print(f"   Training accuracy: {train_acc:.4f}")

    return pipeline


def save_model(pipeline: Pipeline, path: str = MODEL_SAVE_PATH):
    """
    Save the trained pipeline to disk using joblib.

    Parameters
    ----------
    pipeline : Pipeline
        The fitted sklearn Pipeline.
    path : str
        Output file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"\n💾 Model saved to: {path}")


def main():
    """
    Execute the full training pipeline.

    Workflow:
    1. Load data → 2. Inspect → 3. Clean → 4. Split →
    5. Build pipeline → 6. Train → 7. Evaluate → 8. Save
    """
    print("=" * 60)
    print("  Predictive Sales Analytics Engine — Training Pipeline")
    print("=" * 60)

    # Step 1: Load data
    print("\n📂 Step 1: Loading data...")
    df = load_data(DATA_DIR)

    # Step 2: Inspect data
    print("\n🔍 Step 2: Inspecting data...")
    inspection = inspect_data(df)

    # Step 3: Handle missing values
    print("\n🧹 Step 3: Handling missing values...")
    df = handle_missing_values(df)

    # Step 4: Train-test split
    print("\n✂️  Step 4: Splitting data (70/30, stratified)...")
    X_train, X_test, y_train, y_test = split_data(df)

    # Step 5: Build pipeline
    print("\n🔧 Step 5: Building ML pipeline...")
    pipeline = build_pipeline()

    # Step 6: Train
    print("\n🎯 Step 6: Training model...")
    pipeline = train(pipeline, X_train, y_train)

    # Step 7: Evaluate
    print("\n📊 Step 7: Evaluating on test set...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(y_test, y_pred, y_prob)
    print_evaluation_report(metrics)
    plot_confusion_matrix(metrics)
    display_win_probabilities(y_prob)

    # Step 8: Save model
    print("\n💾 Step 8: Saving model...")
    save_model(pipeline)

    print("\n" + "=" * 60)
    print("  ✅ Training pipeline complete!")
    print("=" * 60)

    return pipeline, metrics


if __name__ == "__main__":
    main()
