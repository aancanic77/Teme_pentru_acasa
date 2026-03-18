from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url=os.environ["GROQ_BASE_URL"]
)

try:
    r = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": "test"}]
    )
    print("OK:", r.choices[0].message.content)
except Exception as e:
    print("EROARE:", e)
