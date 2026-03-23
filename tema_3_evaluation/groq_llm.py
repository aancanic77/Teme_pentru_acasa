from deepeval.models.base_model import DeepEvalBaseLLM
from groq import Groq
import os

class GroqDeepEval(DeepEvalBaseLLM):
    def __init__(self, model: str = "llama-3.1-8b-instant"):
        self.model = model
        self.client = Groq(
            api_key=os.environ["GROQ_API_KEY"],
            base_url=os.environ["GROQ_BASE_URL"]
        )

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model