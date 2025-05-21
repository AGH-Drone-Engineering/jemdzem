from google.genai import types
import cv2
import numpy as np


def image_to_part(image: np.ndarray):
    return types.Part.from_bytes(
        data=cv2.imencode(".png", image)[1].tobytes(),
        mime_type="image/png",
    )
