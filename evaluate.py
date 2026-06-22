import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

model = joblib.load("models/model.joblib")
vectorizer = joblib.load("models/vectorizer.joblib")

df = pd.read_csv("data/sentiment_dataset.csv")
X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_vec = vectorizer.transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

train_acc = accuracy_score(y_train, model.predict(X_train_vec))
test_acc  = accuracy_score(y_test,  model.predict(X_test_vec))

print("=" * 40)
print(f"学習データの精度 : {train_acc:.2%}")
print(f"テストデータの精度: {test_acc:.2%}")
print("=" * 40)

diff = train_acc - test_acc
if diff > 0.1:
    print(f"⚠️  過学習の可能性あり（差: {diff:.2%}）")
else:
    print(f"✅  過学習なし（差: {diff:.2%}）")

print("\n--- テストデータの詳細 ---")
print(classification_report(y_test, model.predict(X_test_vec),
                             target_names=["ネガティブ", "ポジティブ"]))