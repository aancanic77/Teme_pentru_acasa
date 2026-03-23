from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url=os.environ["GROQ_BASE_URL"]
)

try:
    r = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "test"}]
    )
    print("OK:", r.choices[0].message.content)
except Exception as e:
    print("EROARE:", e)
