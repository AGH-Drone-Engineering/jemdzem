"""Example script calling the ``/qa`` API."""

import requests
import cv2
import os


if __name__ == "__main__":
    image = cv2.imread(os.path.join(os.path.dirname(__file__), "YUN_0071.JPG"))

    _, img_encoded = cv2.imencode(".png", image)
    img_bytes = img_encoded.tobytes()

    response = requests.post(
        "http://localhost:8000/qa",
        headers={"X-API-Key": "tym_razem_to_musi_poleciec"},
        files={"file": ("image.png", img_bytes, "image/png")},
        data={
            "question": "How many manequins are there? How many of them are weraing yellow helmets and yellow or red reflective vests? Give me coordinates where they are on picture in form of list"
        },
    )
    print(response.json())
