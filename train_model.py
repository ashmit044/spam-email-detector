import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import joblib

df=pd.read_csv("dataset/spam.csv",sep="\t",names=["label","message"])
df["label"]=df["label"].map({"ham":0,"spam":1})

vectorizer=TfidfVectorizer(stop_words="english")
X=vectorizer.fit_transform(df["message"])
y=df["label"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearSVC()
model.fit(X_train,y_train)

pred=model.predict(X_test)
print("Accuracy:",accuracy_score(y_test,pred))

joblib.dump(model,"model.pkl")
joblib.dump(vectorizer,"vectorizer.pkl")
print("Model saved.")
