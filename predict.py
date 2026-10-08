"""CLI prediction script for Spam Email Detector."""

from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"

def load_artifacts():
    """Load pre-trained model and vectorizer."""
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    except FileNotFoundError:
        print("Error: Model files not found! Please run 'python train_model.py' first.")
        exit(1)

def classify_message(text: str, model, vectorizer):
    """Classify input text message as Spam or Ham."""
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    label = "Spam" if prediction == 1 else "Ham (Legitimate)"
    confidence = probabilities[prediction] * 100
    return label, confidence

def main():
    print("--- 📧 Spam Email & SMS Classifier CLI ---")
    model, vectorizer = load_artifacts()
    print("Model loaded successfully. Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("Enter email/message text: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        if not user_input:
            continue

        label, confidence = classify_message(user_input, model, vectorizer)
        print(f"Result: [{label}] (Confidence: {confidence:.1f}%)\n")

if __name__ == "__main__":
    main()
