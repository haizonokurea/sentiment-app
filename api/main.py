from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        messages=[
            {
                "role": "user",
                "content": f"次の日本語テキストの感情を判定してください。「ポジティブ」か「ネガティブ」の一言だけ答えてください。\n\nテキスト：{input.text}"
            }
        ]
    )
    sentiment = message.content[0].text.strip()
    label = 1 if "ポジティブ" in sentiment else 0
    return {"text": input.text, "label": label, "sentiment": sentiment}