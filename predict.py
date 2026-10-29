"""CLI prediction script for Spam Email Detector."""

from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"


class ModelArtifactsError(RuntimeError):
    """Model artifacts are missing, corrupt, or incompatible with this runtime."""


def load_artifacts(model_path=MODEL_PATH, vectorizer_path=VECTORIZER_PATH):
    """Load model artifacts or raise an actionable error for the caller to show."""
    missing = [
        str(path)
        for path in (model_path, vectorizer_path)
        if not Path(path).is_file()
    ]
    if missing:
        raise ModelArtifactsError(
            "Model files not found: " + ", ".join(missing) + ". Run 'python train_model.py' first."
        )
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    except Exception as error:
        raise ModelArtifactsError(
            "Could not load the trained model files. They may be corrupt or "
            "incompatible; run 'python train_model.py' to recreate them."
        ) from error

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
    try:
        model, vectorizer = load_artifacts()
    except ModelArtifactsError as error:
        print(f"Error: {error}")
        return 1
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

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
