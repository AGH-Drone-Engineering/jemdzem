"""Example script calling the ``/qa`` API."""

import requests
import cv2
import os


if __name__ == "__main__":
    image = cv2.imread(
        os.path.join(os.path.dirname(__file__), "inspekcja/manekin_bez_kamzy.png")
    )

    _, img_encoded = cv2.imencode(".png", image)
    img_bytes = img_encoded.tobytes()

    response = requests.post(
        "http://localhost:8000/qa",
        headers={"X-API-Key": "tym_razem_to_musi_poleciec"},
        files={"file": ("image.png", img_bytes, "image/png")},
        data={"question": "Is there yellow helmet and reflective vest?"},
    )
    print(response.json())
