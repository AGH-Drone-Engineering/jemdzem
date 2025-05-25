"""OCR wrapper around Gemini models."""

import numpy as np
import json
from google.genai import types

from .client import client
from .utils import image_to_part


PROMPT = \
"""
Instructions:

You are an optical character recognition expert. You are given an image of a text. Your task is to extract the text from the image.

Example response:

```json
{
  "text": "The text from the image or empty string if there is no text"
}
```
""".strip()


class GeminiOCR:
    """Simple OCR helper using a Gemini model."""

    def __init__(self) -> None:
        self.model_name = "gemini-2.0-flash"

    def ocr(self, image: np.ndarray) -> str:
        """Return recognized text from ``image``."""

        contents = [
            types.Content(
                role="user",
                parts=[image_to_part(image), types.Part.from_text(text=PROMPT)],
            )
        ]

        resp = client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(response_mime_type="text/plain"),
        )
        return json.loads(resp.text.removeprefix("```json").removesuffix("```"))["text"]
