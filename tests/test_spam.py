"""Unit tests for Spam Email Detector."""

import pytest
import joblib
import app
import train_model as training
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from train_model import train_model
from predict import ModelArtifactsError, classify_message, load_artifacts


def test_artifact_loader_reports_missing_files(tmp_path):
    with pytest.raises(ModelArtifactsError, match="Run 'python train_model.py'"):
        load_artifacts(tmp_path / "model.pkl", tmp_path / "vectorizer.pkl")


def test_artifact_loader_reports_corrupt_files(tmp_path):
    model_path = tmp_path / "model.pkl"
    vectorizer_path = tmp_path / "vectorizer.pkl"
    model_path.write_bytes(b"not a joblib artifact")
    vectorizer_path.write_bytes(b"not a joblib artifact")

    with pytest.raises(ModelArtifactsError, match="corrupt or incompatible"):
        load_artifacts(model_path, vectorizer_path)

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


def test_evaluation_vectorizer_does_not_fit_on_held_out_messages(tmp_path, monkeypatch):
    data_path = tmp_path / "messages.tsv"
    data_path.write_text(
        "ham\ttraining hamtoken\n"
        "spam\ttraining spamtoken\n"
        "ham\tsecond hamtoken\n"
        "spam\tsecond spamtoken\n"
        "ham\tthird hamtoken\n"
        "spam\tthird spamtoken\n"
        "ham\tholdoutonlyham hamtoken\n"
        "spam\tholdoutonlyspam spamtoken\n",
        encoding="utf-8",
    )

    def fixed_split(messages, labels, **_kwargs):
        return (
            messages.iloc[:6],
            messages.iloc[6:],
            labels.iloc[:6],
            labels.iloc[6:],
        )

    monkeypatch.setattr(training, "train_test_split", fixed_split)
    fitted_vocabularies = []
    vectorizer_class = training.TfidfVectorizer
    original_fit_transform = vectorizer_class.fit_transform

    def track_fit_transform(self, raw_documents, y=None):
        matrix = original_fit_transform(self, raw_documents, y)
        fitted_vocabularies.append(set(self.vocabulary_))
        return matrix

    monkeypatch.setattr(vectorizer_class, "fit_transform", track_fit_transform)
    train_model(
        data_path=data_path,
        model_path=tmp_path / "model.pkl",
        vectorizer_path=tmp_path / "vectorizer.pkl",
    )

    assert "holdoutonlyham" not in fitted_vocabularies[0]
    assert "holdoutonlyspam" not in fitted_vocabularies[0]
    assert {"holdoutonlyham", "holdoutonlyspam"} <= fitted_vocabularies[1]


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
