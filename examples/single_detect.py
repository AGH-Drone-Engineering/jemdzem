"""Example script calling the ``/single-detect`` API and visualising results."""

import requests
import matplotlib.pyplot as plt
import cv2
import os
import json


if __name__ == "__main__":
    image = cv2.imread(os.path.join(os.path.dirname(__file__), "detect.jpeg"))

    _, img_encoded = cv2.imencode(".png", image)
    img_bytes = img_encoded.tobytes()

    ref_image = cv2.imread(os.path.join(os.path.dirname(__file__), "keys.png"))

    _, ref_img_encoded = cv2.imencode(".png", ref_image)
    ref_img_bytes = ref_img_encoded.tobytes()

    data = {
        "labels": json.dumps(["woman", "keys"]),
        "descriptions": json.dumps(["a woman", "car keys"]),
    }

    response = requests.post(
        "http://localhost:8000/single-detect?model_name=gemini-2.5-flash-preview-04-17",
        headers={"X-API-Key": "tym_razem_to_musi_poleciec"},
        files=[
            ("file", ("image.png", img_bytes, "image/png")),
            ("ref_file", ("keys.png", ref_img_bytes, "image/png")),
        ],
        data=data,
    )
    detections = response.json()
    for detection in detections:
        x = int(detection["x"] * image.shape[1])
        y = int(detection["y"] * image.shape[0])
        width = int(detection["width"] * image.shape[1])
        height = int(detection["height"] * image.shape[0])
        color = {
            "woman": (0, 255, 0),
            "keys": (255, 0, 255),
        }[detection["label"]]
        cv2.rectangle(image, (x, y), (x + width, y + height), color, 8)
        cv2.putText(
            image, detection["label"], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 8
        )
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
