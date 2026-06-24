from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from janome.tokenizer import Tokenizer
import joblib
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

t = Tokenizer()
model = joblib.load(os.path.join(os.path.dirname(__file__), "../models/model.joblib"))
vectorizer = joblib.load(os.path.join(os.path.dirname(__file__), "../models/vectorizer.joblib"))

def tokenize(text):
    tokens = t.tokenize(text)
    return " ".join([token.reading if token.reading != '*' else token.surface for token in tokens])

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    tokenized = tokenize(input.text)
    X_vec = vectorizer.transform([tokenized])
    label = int(model.predict(X_vec)[0])
    sentiment = "ポジティブ" if label == 1 else "ネガティブ"
    return {"text": input.text, "label": label, "sentiment": sentiment}