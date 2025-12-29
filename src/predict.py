import joblib
import os

MODEL_PATH = os.path.join('models', 'phishing_model.pkl')


def load_model():
    if not os.path.exists(MODEL_PATH):
        print("Error: Model not found. Please run Option 1 (Train) first.")
        return None
    return joblib.load(MODEL_PATH)


def predict_single(text):
    """
    Standard prediction for any raw text string.
    """
    model = load_model()
    if not model:
        return "Error", 0.0

    # Predict
    # The pipeline in the model handles the cleaning automatically!
    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0]

    # Logic: 1 = Phishing, 0 = Safe
    label = "PHISHING" if pred == 1 else "Legitimate"
    confidence = proba[1] if pred == 1 else proba[0]

    return label, confidence


def predict_email(subject, body):
    """
    Combines Subject and Body for the model.
    """
    full_text = f"{subject} {body}"
    return predict_single(full_text)


def interactive_mode():
    """
    Detailed Interactive Mode (Option 2).
    Asks the user what specific type of channel they are testing.
    """
    print("\n--- Interactive Test Mode ---")
    print("Type 'exit' to return to main menu.\n")

    while True:
        print("\nSelect Input Type:")
        print("1. Email (Input Subject & Body separately)")
        print("2. SMS / URL (Input single text)")
        choice = input("Choice (1/2): ").strip()

        if choice == '1':
            sub = input("   Enter Subject: ")
            if sub.lower() == 'exit': break
            body = input("   Enter Body:    ")

            label, conf = predict_email(sub, body)
            print(f"   >>> Result: {label} (Confidence: {conf:.1%})")

        elif choice == '2':
            text = input("   Enter Message/URL: ")
            if text.lower() == 'exit': break

            label, conf = predict_single(text)
            print(f"   >>> Result: {label} (Confidence: {conf:.1%})")

        elif choice.lower() == 'exit':
            break
        else:
            print("Invalid choice. Try again.")