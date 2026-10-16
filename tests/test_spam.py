"""Unit tests for Spam Email Detector."""

import pytest
import joblib
import app
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
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


def test_streamlit_loads_artifacts_from_app_directory(tmp_path, monkeypatch):
    app_dir = tmp_path / "app"
    launch_dir = tmp_path / "elsewhere"
    app_dir.mkdir()
    launch_dir.mkdir()

    vectorizer = TfidfVectorizer().fit(["spam offer", "hello friend"])
    features = vectorizer.transform(["spam offer", "hello friend"])
    model = MultinomialNB().fit(features, [1, 0])
    model_path = app_dir / "model.pkl"
    vectorizer_path = app_dir / "vectorizer.pkl"
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    monkeypatch.setattr(app, "MODEL_PATH", model_path)
    monkeypatch.setattr(app, "VECTORIZER_PATH", vectorizer_path)
    monkeypatch.chdir(launch_dir)
    app.load_model.clear()

    loaded_model, loaded_vectorizer = app.load_model()

    assert loaded_model is not None
    assert loaded_vectorizer is not None
    assert loaded_model.predict(loaded_vectorizer.transform(["spam offer"]))[0] == 1
