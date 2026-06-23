from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
import MeCab

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, "models", "model.joblib"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "models", "vectorizer.joblib"))

tagger = MeCab.Tagger("-r /etc/mecabrc")

def tokenize(text):
    node = tagger.parseToNode(text)
    words = []
    while node:
        if node.surface:
            words.append(node.surface)
        node = node.next
    return " ".join(words)

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    tokenized = tokenize(input.text)
    X = vectorizer.transform([tokenized])
    pred = model.predict(X)[0]
    label = "ポジティブ" if pred == 1 else "ネガティブ"
    return {"text": input.text, "label": int(pred), "sentiment": label}