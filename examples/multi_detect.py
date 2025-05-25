"""Example script calling the ``/multi-detect`` API and visualising results."""

import requests
import matplotlib.pyplot as plt
import cv2
import os
import json


if __name__ == "__main__":
    image = cv2.imread(os.path.join(os.path.dirname(__file__), "detect.jpeg"))

    _, img_encoded = cv2.imencode(".png", image)
    img_bytes = img_encoded.tobytes()

    data = {
        "labels": json.dumps(["man", "woman", "car", "keys"]),
        "descriptions": json.dumps(
            ["A man", "A woman", "A car", "Keys, car keys or a keychain"]
        ),
    }

    response = requests.post(
        "http://localhost:8000/multi-detect",
        headers={"X-API-Key": "tym_razem_to_musi_poleciec"},
        files={"file": ("image.png", img_bytes, "image/png")},
        data=data,
    )
    detections = response.json()
    for detection in detections:
        x = int(detection["x"] * image.shape[1])
        y = int(detection["y"] * image.shape[0])
        width = int(detection["width"] * image.shape[1])
        height = int(detection["height"] * image.shape[0])
        color = {
            "man": (0, 0, 255),
            "woman": (0, 255, 0),
            "car": (255, 0, 0),
            "keys": (255, 0, 255),
        }[detection["label"]]
        cv2.rectangle(image, (x, y), (x + width, y + height), color, 8)
        cv2.putText(
            image, detection["label"], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 8
        )
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
