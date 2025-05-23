import json
import numpy as np
from google.genai import types

from .client import client
from .utils import image_to_part


PROMPT = \
"""
Instructions:

Target object: {{TARGET_OBJECT}}
Object description: {{OBJECT_DESCRIPTION}}
{{EXTRA_INSTRUCTIONS}}
You are an object detection expert. Analyze the image and locate instances of the object. For each object, include:
1. box_2d: The bounding box in [ymin, xmin, ymax, xmax] format.

Output the analysis in JSON format like this:

```json
[
{
    "box_2d": [ymin, xmin, ymax, xmax]
}
]
```

Example Output:

```json
[
{
    "box_2d": [344, 797, 462, 998]
},
{
    "box_2d": [152, 456, 56, 98]
}
]
```
""".strip()


class GeminiSingleDetector:
    def detect(self, image: np.ndarray, label: str, description: str, model_name: str, ref_image: np.ndarray | None = None) -> list[dict]:
        prompt = PROMPT.replace("{{TARGET_OBJECT}}", label).replace("{{OBJECT_DESCRIPTION}}", description)
        if ref_image is not None:
            prompt = prompt.replace("{{EXTRA_INSTRUCTIONS}}", "A reference image of the object is provided as the first image. You need to find this object in the second image.\n")
        contents = [
            types.Content(
                role="user",
                parts=[
                    image_to_part(image),
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]
        if ref_image is not None:
            contents[0].parts.insert(0, image_to_part(ref_image))
        resp = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="text/plain",
            ),
        )
        boxes = json.loads(resp.text.removeprefix("```json").removesuffix("```"))
        return [{
            "x": box["box_2d"][1] / 1000,
            "y": box["box_2d"][0] / 1000,
            "width": (box["box_2d"][3] - box["box_2d"][1]) / 1000,
            "height": (box["box_2d"][2] - box["box_2d"][0]) / 1000,
        } for box in boxes]
