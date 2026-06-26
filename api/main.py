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

# 単語辞書
word_dict = {
    "好き": 1, "すき": 1, "良い": 1, "よい": 1, "いい": 1,
    "楽しい": 1, "たのしい": 1, "嬉しい": 1, "うれしい": 1,
    "面白い": 1, "おもしろい": 1, "素晴らしい": 1, "すばらしい": 1,
    "最高": 1, "さいこう": 1, "感動": 1, "かんどう": 1,
    "満足": 1, "まんぞく": 1, "便利": 1, "べんり": 1,
    "可愛い": 1, "かわいい": 1, "綺麗": 1, "きれい": 1,
    "神": 1, "完璧": 1, "かんぺき": 1, "大好き": 1, "だいすき": 1,
    "幸せ": 1, "しあわせ": 1, "おすすめ": 1, "最強": 1,
    "100点": 1, "１００点": 1, "満点": 1, "パーフェクト": 1,
    "嫌い": 0, "きらい": 0, "悪い": 0, "わるい": 0,
    "つまらない": 0, "悲しい": 0, "かなしい": 0, "辛い": 0, "つらい": 0,
    "最悪": 0, "さいあく": 0, "最低": 0, "さいてい": 0,
    "ひどい": 0, "使いにくい": 0, "つかいにくい": 0, "使いづらい": 0,
    "うざい": 0, "きしょい": 0, "後悔": 0, "こうかい": 0,
    "不満": 0, "ふまん": 0, "失望": 0, "しつぼう": 0,
    "ゴミ": 0, "クソ": 0, "最弱": 0, "痛い": 0, "いたい": 0,
    "うんこ": 0, "ウンコ": 0, "うんち": 0, "ウンチ": 0,
}

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