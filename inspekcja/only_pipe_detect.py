"""Example script calling the ``/single-detect`` API and visualising results."""

import requests
import matplotlib.pyplot as plt
import cv2
import os
import json


if __name__ == "__main__":
    # objects = ["pipe","powerpole","barrell","palette","person","car"]
    objects = ["pipe"]

    """filepaths = ["inspekcja/rura_urwana.JPG",
    "inspekcja/stojak.jpg",
    "inspekcja/beczka.png", 
    #"inspekcja/paleta.png"
    ]"""

    filepaths = ["rura_urwana.JPG"]

    image = cv2.imread(os.path.join(os.path.dirname(__file__), "YUN_0155.JPG"))
    _, img_encoded = cv2.imencode(".png", image)
    img_bytes = img_encoded.tobytes()

    pictures = [("file", ("image.png", img_bytes, "image/png"))]

    i = 0
    for path in filepaths:
        ref_image = cv2.imread(os.path.join(os.path.dirname(__file__), path))

        _, ref_img_encoded = cv2.imencode(".png", ref_image)
        ref_img_bytes = ref_img_encoded.tobytes()
        name = "ref_file" + str(i)
        pictures.append((name, (path, ref_img_bytes, "image/png")))
        i += 1

        """["find all oragne pipes",
                                    "find black powerpoles and do not confuse them with shadows",
                                    "find all blue barrels",
                                    "find all wooden palettes",
                                    "find all people",
                                    "find all cars"]"""

    data = {
        "labels": json.dumps(["pipe"]),
        "descriptions": json.dumps(["find oragne pipes"]),
    }

    response = requests.post(
        "http://localhost:8000/single-detect?model_name=gemini-2.5-flash-preview-04-17",
        headers={"X-API-Key": "tym_razem_to_musi_poleciec"},
        files=[
            ("file", ("image.png", img_bytes, "image/png")),
            ("ref_file", ("rura_urwana.JPG", ref_img_bytes, "image/png")),
            # ("ref_file", ("stojak.jpeg", ref_img_bytes2, "image/png")),
        ],
        data=data,
    )

    print(response)

    detections = response.json()
    for detection in detections:
        x = int(detection["x"] * image.shape[1])
        y = int(detection["y"] * image.shape[0])
        width = int(detection["width"] * image.shape[1])
        height = int(detection["height"] * image.shape[0])
        color = {
            "pipe": (0, 255, 0),
            "powerpole": (255, 0, 255),
            "barrell": (255, 255, 0),
            "palette": (0, 255, 255),
            "person": (255, 0, 0),
            "car": (0, 0, 255),
        }[detection["label"]]
        cv2.rectangle(image, (x, y), (x + width, y + height), color, 8)
        cv2.putText(
            image, detection["label"], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 8
        )
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.savefig("plot.png")
