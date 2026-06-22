import requests

text = input("テキストを入力してください: ")

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"text": text}
)
data = response.json()

match data["label"]:
    case 1:
        print(f"😊 ポジティブ（label: 1）")
    case 0:
        print(f"😞 ネガティブ（label: 0）")
    case _:
        print("エラー: 予期しない結果が返ってきました")