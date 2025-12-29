import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
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
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',  # <--- ADD THIS LINE
            n_jobs=-1,
            random_state=42
        ))
    ])

    # 4. Train
    print("Training started...")
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    # 5. Evaluate
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nModel Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, preds))

    # 6. Save Model
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f" Model saved to {MODEL_PATH}")

    # 7. Generate Confusion Matrix (For your report)
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('confusion_matrix.png')
    print(" Confusion matrix saved as 'confusion_matrix.png'")