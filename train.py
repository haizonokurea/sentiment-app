import pandas as pd
import joblib
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

t = Tokenizer()

def tokenize(text):
    tokens = t.tokenize(text)
    return " ".join([token.reading if token.reading != '*' else token.surface for token in tokens])

# データ読み込み
df = pd.read_csv("data/sentiment_dataset.csv")
X = df["text"].apply(tokenize)
y = df["label"]

# モデル学習
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vec, y)

# 保存
joblib.dump(model, "models/model.joblib")
joblib.dump(vectorizer, "models/vectorizer.joblib")

print("学習完了！")