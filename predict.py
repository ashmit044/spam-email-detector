import joblib

model=joblib.load("model.pkl")
vectorizer=joblib.load("vectorizer.pkl")

while True:
    msg=input("Enter message (exit to quit): ")
    if msg.lower()=="exit":
        break
    pred=model.predict(vectorizer.transform([msg]))[0]
    print("Spam" if pred==1 else "Ham")
