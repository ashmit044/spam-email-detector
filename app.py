"""Streamlit Web Dashboard for Spam Email Detector."""

import streamlit as st
import joblib
import os

st.set_page_config(
    page_title="Spam Email & Message Classifier",
    page_icon="📧",
    layout="centered"
)

@st.cache_resource
def load_model():
    """Load model and vectorizer with caching for performance."""
    if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
        return None, None
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

def main():
    st.title("📧 Spam Email & SMS Detector")
    st.write("A beginner machine learning app built with **Scikit-Learn** and **Streamlit** to detect spam messages using Naive Bayes classification.")

    model, vectorizer = load_model()

    if model is None or vectorizer is None:
        st.warning("⚠️ Pre-trained model files not found. Please train the model first by running `python train_model.py` in your terminal.")
        return

    st.markdown("---")
    st.subheader("🔍 Classify a Message")

    # Sample Preset Selection
    sample_choice = st.selectbox(
        "Or pick a sample message to test:",
        [
            "Custom Input",
            "URGENT! You have won a $1,000 Walmart gift card! Claim now at http://bit.ly/fake123",
            "Hey, are we still meeting for lunch at 1 PM today?",
            "CONGRATULATIONS! You have been selected for a free iPhone 15. Call 0800-SPAM now!",
            "Hi Mom, I left my keys at home. Can you call me back when you get this?"
        ]
    )

    if sample_choice != "Custom Input":
        default_text = sample_choice
    else:
        default_text = ""

    user_text = st.text_area("Enter Message Text:", value=default_text, height=120, placeholder="Paste email or SMS content here...")

    if st.button("Check Message", type="primary") or user_text:
        if user_text.strip():
            features = vectorizer.transform([user_text])
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]

            spam_prob = probabilities[1] * 100
            ham_prob = probabilities[0] * 100

            st.markdown("### 📊 Result")
            if prediction == 1:
                st.error(f"🚨 **SPAM DETECTED** (Probability: {spam_prob:.1f}%)")
            else:
                st.success(f"✅ **HAM / LEGITIMATE MESSAGE** (Probability: {ham_prob:.1f}%)")

            st.progress(int(spam_prob))
            st.caption(f"Spam probability: {spam_prob:.1f}% | Ham probability: {ham_prob:.1f}%")
        else:
            st.info("Please enter a text message to classify.")

if __name__ == "__main__":
    main()
