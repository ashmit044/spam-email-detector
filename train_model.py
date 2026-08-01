"""Train Spam Detection Model using Scikit-Learn (TF-IDF + Naive Bayes/SVM).

This script reads spam.csv dataset, preprocesses text using TF-IDF,
trains a classifier model, evaluates performance, and saves the trained artifacts.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

def main():
    print("--- Training Spam Detector Model ---")

    # 1. Load dataset (SMS / Email spam dataset)
    # The dataset uses tab-separated values with columns: label ('ham' or 'spam') and message text
    df = pd.read_csv("spam.csv", sep="\t", names=["label", "message"])
    print(f"Loaded {len(df)} records from dataset.")

    # 2. Encode target labels: 0 for Ham (Legitimate), 1 for Spam
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

    # 3. Vectorize text messages using TF-IDF (Term Frequency - Inverse Document Frequency)
    # TF-IDF converts raw text strings into numerical feature matrices based on word importance.
    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
    X = vectorizer.fit_transform(df["message"])
    y = df["label_num"]

    # 4. Split dataset into training set (80%) and test set (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

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

    # 7. Save model and vectorizer to disk using joblib
    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    print("Successfully saved 'model.pkl' and 'vectorizer.pkl'.")

if __name__ == "__main__":
    main()
