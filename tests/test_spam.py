"""Unit tests for Spam Email Detector."""

import pytest
import joblib
from train_model import main as train_main
from predict import MODEL_PATH, VECTORIZER_PATH, classify_message

def test_model_training_and_prediction():
    # Execute training script to generate artifacts
    train_main()

    assert MODEL_PATH.exists()
    assert VECTORIZER_PATH.exists()

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

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
