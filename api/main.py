from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from janome.tokenizer import Tokenizer
import joblib
import pandas as pd
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

# 単語ラベルCSVを読み込む
word_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/word_labels.csv"))
word_dict = dict(zip(word_df["単語"], word_df["ラベル"]))

def tokenize(text):
    tokens = t.tokenize(text)
    return " ".join([token.reading if token.reading != '*' else token.surface for token in tokens])

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    tokens = t.tokenize(input.text)
    score = 0
    count = 0
    for token in tokens:
        for w in [token.surface, token.reading]:
            if w in word_dict:
                score += word_dict[w]
                count += 1

    if count == 0:
        tokenized = tokenize(input.text)
        X_vec = vectorizer.transform([tokenized])
        label = int(model.predict(X_vec)[0])
    else:
        label = 1 if score / count >= 0.5 else 0

    sentiment = "ポジティブ" if label == 1 else "ネガティブ"
    return {"text": input.text, "label": label, "sentiment": sentiment}