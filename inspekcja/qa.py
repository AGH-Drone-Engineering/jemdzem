"""Example script calling the ``/qa`` API."""

import requests
import cv2
import os


if __name__ == "__main__":
    image = cv2.imread(os.path.join(os.path.dirname(__file__), "YUN_0195.JPG"))

    _, img_encoded = cv2.imencode(".png", image)
    img_bytes = img_encoded.tobytes()
    
    image2 = cv2.imread(os.path.join(os.path.dirname(__file__), "YUN_0155b.jpg"))

    _, img_encoded2 = cv2.imencode(".png", image2)
    img_bytes2 = img_encoded2.tobytes()

    response = requests.post(
        "http://localhost:8000/qa",
        headers={"X-API-Key": "tym_razem_to_musi_poleciec"},
        files={"file": ("image.png", img_bytes, "image/png"),
                "second_file": ("image2.png", img_bytes2, "image2/png")},
        data={
            "question": "What are the differences between these two pictures?"
        },
    )
    print(response.json())
