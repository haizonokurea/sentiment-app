from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI()

# CORSの設定を追加
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, "models", "model.joblib"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "models", "vectorizer.joblib"))

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    X = vectorizer.transform([input.text])
    pred = model.predict(X)[0]
    label = "ポジティブ" if pred == 1 else "ネガティブ"
    return {"text": input.text, "label": int(pred), "sentiment": label}