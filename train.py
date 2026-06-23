import pandas as pd
import joblib
import MeCab
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# MeCabで単語に分割する関数
tagger = MeCab.Tagger("-r /etc/mecabrc")

def tokenize(text):
    node = tagger.parseToNode(text)
    words = []
    while node:
        if node.surface:
            words.append(node.surface)
        node = node.next
    return " ".join(words)

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