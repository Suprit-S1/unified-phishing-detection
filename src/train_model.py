import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from src.data_loader import load_and_unify_data
from src.features import build_feature_extractor

MODEL_PATH = os.path.join('models', 'phishing_model.pkl')


def train():

    # 1. Load Data
    data = load_and_unify_data()
    X = data['text'].astype(str)
    y = data['label'].astype(int)

    # 2. Split
    print("Splitting data (80% Train / 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Build Pipeline
    print("Building Random Forest Pipeline...")
    pipeline = Pipeline([
        ('features', build_feature_extractor()),
        ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=42))
    ])

    # 4. Train
    print("Training started...")
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    # 5. Evaluate
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nModel Accuracy: {acc * 100:.2f}%")

    # 6. Save Model
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f" Model saved to {MODEL_PATH}")

    # Get feature names from the TF-IDF vectorizer
    # Note: We access the step by the name string 'features' defined in the pipeline above
    feature_names = pipeline.named_steps['features'].get_feature_names_out()

    # Get feature importances from the Random Forest
    # Note: We access the step by the name string 'classifier' defined above
    importances = pipeline.named_steps['classifier'].feature_importances_

    # Sort them
    indices = np.argsort(importances)[-20:]  # Top 20 features

    # Plot
    plt.figure(figsize=(10, 8))
    plt.title("Top 20 Most Important Signals for Phishing Detection")
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Gini Importance Score")
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Feature importance plot saved.")


if __name__ == "__main__":
    train()