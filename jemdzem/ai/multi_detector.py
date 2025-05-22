import json
import numpy as np
from google.genai import types

from .client import client
from .utils import image_to_part


PROMPT = \
"""
Instructions:

You are given a list of objects with their descriptions (label: description).
{{OBJECTS}}

You are an object detection expert. Analyze the image and locate all objects. For each object, include:
1. label: The label of the object.
2. box_2d: The bounding box in [ymin, xmin, ymax, xmax] format.

Output the analysis in JSON format like this:

```json
[
{
    "label": "object_label",
    "box_2d": [ymin, xmin, ymax, xmax]
}
]
```

Example Output:

```json
[
{
    "label": "car",
    "box_2d": [344, 797, 462, 998]
},
{
    "label": "person",
    "box_2d": [152, 456, 56, 98]
}
]
```
""".strip()


class GeminiMultiDetector:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def detect(self, image: np.ndarray, labels: list[str], descriptions: list[str]) -> list[dict]:
        prompt = PROMPT.replace(
            "{{OBJECTS}}",
            "\n".join([f"{label}: {description}" for label, description in zip(labels, descriptions)]))
        contents = [
            types.Content(
                role="user",
                parts=[
                    image_to_part(image),
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]
        resp = client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="text/plain",
            ),
        )
        boxes = json.loads(resp.text.removeprefix("```json").removesuffix("```"))
        return [{
            "label": box["label"],
            "x": box["box_2d"][1] / 1000,
            "y": box["box_2d"][0] / 1000,
            "width": (box["box_2d"][3] - box["box_2d"][1]) / 1000,
            "height": (box["box_2d"][2] - box["box_2d"][0]) / 1000,
        } for box in boxes]
