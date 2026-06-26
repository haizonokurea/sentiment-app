from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from janome.tokenizer import Tokenizer
import joblib
import oseti
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

t = Tokenizer()
analyzer = oseti.Analyzer()
model = joblib.load(os.path.join(os.path.dirname(__file__), "../models/model.joblib"))
vectorizer = joblib.load(os.path.join(os.path.dirname(__file__), "../models/vectorizer.joblib"))

def tokenize(text):
    tokens = t.tokenize(text)
    return " ".join([token.reading if token.reading != '*' else token.surface for token in tokens])

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    # 自分たちのモデルで判定
    tokenized = tokenize(input.text)
    X_vec = vectorizer.transform([tokenized])
    ml_label = int(model.predict(X_vec)[0])
    ml_proba = model.predict_proba(X_vec)[0]
    ml_confidence = float(max(ml_proba))

    # osetiで判定
    oseti_scores = analyzer.analyze(input.text)
    if oseti_scores:
        oseti_score = sum(oseti_scores) / len(oseti_scores)
        oseti_label = 1 if oseti_score > 0 else 0
    else:
        oseti_label = ml_label
        oseti_score = 0

    # 信頼度が高い場合はMLモデル、低い場合はosetiを優先
    if ml_confidence >= 0.7:
        label = ml_label
    else:
        label = oseti_label

    sentiment = "ポジティブ" if label == 1 else "ネガティブ"
    return {"text": input.text, "label": label, "sentiment": sentiment}