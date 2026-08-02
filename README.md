# 📧 Spam Email & SMS Detector

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?style=flat-square&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/Streamlit-1.25.0-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License" />
</p>

A beginner-friendly Natural Language Processing (NLP) machine learning project that classifies text messages and emails as **Spam** or **Ham (Legitimate)** using TF-IDF feature extraction and Naive Bayes classification.

I built this project as a Computer Science (AI & ML) student to explore foundational NLP text preprocessing, vectorization, and model deployment using Streamlit.

---

## ✨ Features

- 🧹 **TF-IDF Text Vectorization**: Converts raw text strings into numerical feature matrices using Term Frequency-Inverse Document Frequency.
- 🤖 **Multinomial Naive Bayes Model**: Fast probabilistic text classification suited for document classification.
- 🌐 **Streamlit Web Dashboard**: Interactive web app to test custom text or pre-loaded sample messages.
- 💻 **CLI Prediction Mode**: Quick command-line interface for testing messages directly in terminal.
- 🧪 **Unit Tests**: Includes `pytest` unit testing suite for model training and prediction functions.

---

## 🚀 Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/ashmit044/spam-email-detector.git
cd spam-email-detector
```

### 2. Create Virtual Environment & Install Dependencies
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

---

## 🏃 How to Run

### 1. Train the Model
Train the Multinomial Naive Bayes model on `spam.csv`:
```bash
python train_model.py
```
This generates `model.pkl` and `vectorizer.pkl`.

### 2. Launch Streamlit Web UI
```bash
streamlit run app.py
```

### 3. Run CLI Predictor
```bash
python predict.py
```

### 4. Run Unit Tests
```bash
pytest
```

---

## 🛠️ Project Structure

```text
spam-email-detector/
├── spam.csv             # Tab-separated dataset (SMS/Email messages)
├── train_model.py       # Model training, evaluation & serialization script
├── app.py               # Streamlit web dashboard interface
├── predict.py           # Command-line predictor interface
├── tests/               # Unit tests
│   └── test_spam.py
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignore file
└── README.md            # Project documentation
```

---

## 📚 What I Learned

- How **TF-IDF (Term Frequency-Inverse Document Frequency)** highlights informative words while downweighting common stopwords.
- Implementing **Multinomial Naive Bayes** for binary text classification.
- Saving trained models using `joblib` for reusability.
- Building interactive Web UI dashboards using **Streamlit**.

---

## 📄 License

Distributed under the MIT License.
