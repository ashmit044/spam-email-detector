"""Unit tests for Spam Email Detector."""

import pytest
import joblib
from train_model import train_model
from predict import classify_message

def test_model_training_and_prediction(tmp_path):
    model_path = tmp_path / "model.pkl"
    vectorizer_path = tmp_path / "vectorizer.pkl"
    train_model(model_path=model_path, vectorizer_path=vectorizer_path)

    assert model_path.exists()
    assert vectorizer_path.exists()

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # Test obvious spam message
    spam_msg = "WINNER! You have won $1000 cash. Call 0800123456 now!"
    label, confidence = classify_message(spam_msg, model, vectorizer)
    assert "Spam" in label
    assert confidence > 50.0

    # Test obvious ham message
    ham_msg = "Hey, are you free for a study session tomorrow at 5?"
    label_ham, conf_ham = classify_message(ham_msg, model, vectorizer)
    assert "Ham" in label_ham
    assert conf_ham > 50.0
