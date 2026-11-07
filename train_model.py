"""Train Spam Detection Model using Scikit-Learn (TF-IDF + Naive Bayes/SVM).

This script reads spam.csv dataset, preprocesses text using TF-IDF,
trains a classifier model, evaluates performance, and saves the trained artifacts.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "spam.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"

def train_model(
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
    vectorizer_path: Path = VECTORIZER_PATH,
):
    """Train the classifier and write artifacts to the requested locations."""
    data_path = Path(data_path)
    model_path = Path(model_path)
    vectorizer_path = Path(vectorizer_path)
    print("--- Training Spam Detector Model ---")

    # 1. Load dataset (SMS / Email spam dataset)
    # The dataset uses tab-separated values with columns: label ('ham' or 'spam') and message text
    df = pd.read_csv(data_path, sep="\t", names=["label", "message"])
    print(f"Loaded {len(df)} records from dataset.")

    # 2. Encode target labels: 0 for Ham (Legitimate), 1 for Spam
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

    y = df["label_num"]

    # Split raw messages before fitting TF-IDF, so evaluation data cannot
    # influence the learned vocabulary or inverse document frequencies.
    messages_train, messages_test, y_train, y_test = train_test_split(
        df["message"], y, test_size=0.2, random_state=42, stratify=y
    )
    evaluation_vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
    X_train = evaluation_vectorizer.fit_transform(messages_train)
    X_test = evaluation_vectorizer.transform(messages_test)

    # 5. Train Multinomial Naive Bayes classifier
    # Naive Bayes is a classic, effective probabilistic algorithm for text classification.
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # 6. Evaluate model on test dataset
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, predictions, target_names=["Ham", "Spam"]))

    # Refit production artifacts on the full dataset after measuring holdout
    # performance so the deployed model uses every labeled example.
    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
    X_all = vectorizer.fit_transform(df["message"])
    production_model = MultinomialNB()
    production_model.fit(X_all, y)

    # 7. Save production model and vectorizer to disk using joblib
    model_path.parent.mkdir(parents=True, exist_ok=True)
    vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(production_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print("Successfully saved 'model.pkl' and 'vectorizer.pkl'.")
    return production_model, vectorizer


def main():
    """Train and save the production model artifacts beside this script."""
    train_model()

if __name__ == "__main__":
    main()
