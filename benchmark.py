import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Import your project's custom modules
from src.data_loader import load_and_unify_data
from src.features import build_feature_extractor

def run_benchmark():
    # --- 1. Load Data (The "Usual" Part) ---
    print("Loading data...")
    # This calls your existing data_loader.py logic
    data = load_and_unify_data()

    X = data['text'].astype(str)
    y = data['label'].astype(int)

    # Split the data just like you do in train_model.py
    print("Splitting data (80% Train / 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- 2. Baseline A: Dummy Classifier ---
    # The "Zero Rule" - always predicts the most frequent class (e.g., "Safe")
    print("\nRunning Dummy Classifier (Baseline)...")
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_acc = dummy.score(X_test, y_test)
    print(f"Result: Baseline (Dummy) Accuracy: {dummy_acc:.2%}")

    # --- 3. Baseline B: Naive Bayes ---
    # Standard text classifier. We need a Pipeline to handle the TF-IDF part.
    print("\nRunning Naive Bayes Classifier...")
    nb_pipeline = Pipeline([
        ('features', build_feature_extractor()),  # Uses your features.py
        ('classifier', MultinomialNB())
    ])
    nb_pipeline.fit(X_train, y_train)
    nb_acc = nb_pipeline.score(X_test, y_test)
    print(f"Result: Baseline (Naive Bayes) Accuracy: {nb_acc:.2%}")

if __name__ == "__main__":
    run_benchmark()