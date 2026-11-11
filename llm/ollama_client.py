

import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()


class OllamaClient:

    def __init__(self):
        self.model = os.getenv(
            "OLLAMA_CHAT_MODEL",
            "llama3.2:3b"
        )

        self.llm = ChatOllama(
            model=self.model,
            temperature=0,
            keep_alive="30m"
        )

    def generate_response(self, prompt):
        response = self.llm.invoke(prompt)
        return response.content

    def generate_structured_response(
        self,
        prompt,
        response_model
    ):
        structured_llm = self.llm.with_structured_output(
            response_model,
            method="json_mode"
        )

        return structured_llm.invoke(prompt)


def get_structured_completion(
    prompt,
    response_model
):
    client = OllamaClient()

    return client.generate_structured_response(
        prompt=prompt,
        response_model=response_model
    )


