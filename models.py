"""Weave-traced Mistral models for lyrics generation."""

import os

import weave
from mistralai import Mistral

from dotenv import load_dotenv

load_dotenv()


class MistralLyricsModel(weave.Model):
    """Baseline lyrics generator using Mistral API with minimal prompting."""

    model_name: str
    system_prompt: str = ""
    temperature: float = 0.9
    max_tokens: int = 2048

    @weave.op()
    def predict(self, genre: str, theme: str) -> dict:
        client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

        user_prompt = f"Write a {genre} song about {theme}."

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        response = client.chat.complete(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        lyrics = response.choices[0].message.content
        return {
            "lyrics": lyrics,
            "genre": genre,
            "theme": theme,
            "model": self.model_name,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
        }


class PromptEngineeredLyricsModel(weave.Model):
    """Enhanced lyrics generator with genre-specific prompting strategies."""

    model_name: str
    system_prompts: dict[str, str] = {}
    structural_instructions: str = ""
    few_shot_examples: dict[str, str] = {}
    temperature: float = 0.9
    max_tokens: int = 2048

    @weave.op()
    def predict(self, genre: str, theme: str) -> dict:
        client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

        system = self.system_prompts.get(genre, "")
        if self.structural_instructions:
            system += "\n\n" + self.structural_instructions

        user_prompt = f"Write a {genre} song about {theme}."
        if genre in self.few_shot_examples:
            user_prompt = (
                f"Here's an example of a great {genre} song:\n\n"
                f"{self.few_shot_examples[genre]}\n\n"
                f"Now, write an original {genre} song about {theme}."
            )

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_prompt})

        response = client.chat.complete(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        lyrics = response.choices[0].message.content
        return {
            "lyrics": lyrics,
            "genre": genre,
            "theme": theme,
            "model": self.model_name,
            "approach": "prompt_engineered",
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
        }
