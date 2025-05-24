"""Utility helpers used across the AI modules."""

from google.genai import types
import cv2
import numpy as np


def image_to_part(image: np.ndarray) -> types.Part:
    """Encode an OpenCV image into a ``genai`` content part."""

    return types.Part.from_bytes(
        data=cv2.imencode(".png", image)[1].tobytes(),
        mime_type="image/png",
    )


def box_to_relative(box_2d: list[int]) -> dict[str, float]:
    """Convert a ``[ymin, xmin, ymax, xmax]`` box in 0-1000 range to ``x/y/width/height`` values between ``0`` and ``1``."""

    ymin, xmin, ymax, xmax = box_2d
    return {
        "x": xmin / 1000,
        "y": ymin / 1000,
        "width": (xmax - xmin) / 1000,
        "height": (ymax - ymin) / 1000,
    }
