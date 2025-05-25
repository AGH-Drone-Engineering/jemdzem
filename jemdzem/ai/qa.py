"""Question answering about images using Gemini models."""

import numpy as np
from google.genai import types

from .client import client
from .utils import image_to_part


PROMPT = """Instructions:\n\nYou are an expert in image understanding. Answer the user's question about the provided image in a single short sentence."""


class GeminiQA:
    """Answer free-form questions about an image using Gemini models."""

    def answer(self, image: np.ndarray, question: str, model_name: str) -> str:
        """Return a short answer to ``question`` about ``image``."""

        contents = [
            types.Content(
                role="user",
                parts=[
                    image_to_part(image),
                    types.Part.from_text(text=f"{PROMPT}\n{question}"),
                ],
            ),
        ]

        resp = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(response_mime_type="text/plain"),
        )
        return resp.text.strip()
