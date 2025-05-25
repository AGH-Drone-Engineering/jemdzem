"""Single-class object detection using Gemini models."""

import json
import numpy as np
from google.genai import types

from .client import client
from .utils import image_to_part, box_to_relative


PROMPT = """
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
    """Detect a single class, optionally using a reference image."""

    def detect(
        self,
        image: np.ndarray,
        label: str,
        description: str,
        model_name: str,
        ref_image: np.ndarray | None = None,
    ) -> list[dict]:
        """Return bounding boxes for ``label`` within ``image``."""

        prompt = (
            PROMPT.replace("{{TARGET_OBJECT}}", label)
            .replace("{{OBJECT_DESCRIPTION}}", description)
            .replace(
                "{{EXTRA_INSTRUCTIONS}}",
                "A reference image of the object is provided as the first image. You need to find this object in the second image.\n"
                if ref_image is not None
                else "",
            )
        )

        contents = [
            types.Content(
                role="user",
                parts=[image_to_part(image), types.Part.from_text(text=prompt)],
            ),
        ]

        if ref_image is not None:
            contents[0].parts.insert(0, image_to_part(ref_image))

        resp = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(response_mime_type="text/plain"),
        )

        boxes = json.loads(resp.text.removeprefix("```json").removesuffix("```"))
        return [box_to_relative(box["box_2d"]) for box in boxes]
