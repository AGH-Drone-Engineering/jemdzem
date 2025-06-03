"""Example script calling the ``/single-detect`` API and visualising results."""

import requests
import matplotlib.pyplot as plt
import cv2
import os
import json
import sys
import raporting.push_point as push_point

if __name__ == "__main__":
    ### ZMIANY RAPORTOWANIE ###
    push_point.clear_points()
    temp_folder = os.path.join(os.path.dirname(__file__), "temp_detections")
    os.makedirs(temp_folder, exist_ok=True)
    ### KONIEC ZMIAN ###
    stream_url = 'rtsp://192.168.241.1/live'

    # Open the video stream
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("Failed to connect to the drone stream.")
        exit()

    ret, image = cap.read()
    if not ret:
        print("Failed to grab frame.")
        exit()

######################## DETECT BARRELL/PIPE/PALETTE ###############
    objects = ["barrell","pipe","palette"]
    filepaths = ["inspekcja/beczka.png","inspekcja/rura_urwana.JPG","paleta.png"]
    descriptions = ["find all blue barrels","find all orange pipes","find all wooden pallettes"]

    #image = cv2.imread(os.path.join(os.path.dirname(__file__), sys.argv[1]))
    _, img_encoded = cv2.imencode(".png", image)
    img_bytes = img_encoded.tobytes()


    for i in range(3):
        ref_image = cv2.imread(os.path.join(os.path.dirname(__file__), filepaths[i]))
        pictures = [("file", ("image.png", img_bytes, "image/png"))]
        _, ref_img_encoded = cv2.imencode(".png", ref_image)
        ref_img_bytes = ref_img_encoded.tobytes()
        name = "ref_file"
        pictures.append((name, (path, ref_img_bytes, "image/png")))

        data = {
            "labels": json.dumps(objects),
            "descriptions": json.dumps(descriptions),
        }

        response = requests.post(
            "http://localhost:8000/single-detect?model_name=gemini-2.5-flash-preview-04-17",
            headers={"X-API-Key": "tym_razem_to_musi_poleciec"},
            files=pictures,
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
                "barrell": (255, 255, 0),
                "palette": (0, 255, 255),
                "person": (255, 0, 0),
                "car": (0, 0, 255),
            }[detection["label"]]
        ### ZMIANY RAPORTOWANIE ###
            push_point.push_detection_to_firebase(detection, (y,x), temp_path)
            ### KONIEC ZMIAN ###
            cv2.rectangle(image, (x, y), (x + width, y + height), color, 8)
            cv2.putText(
                image, detection["label"], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 8
            )
        #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.savefig("plot" + str(i) + ".png")

############################## DETECT PEOPLE #########################################
    data = {
        "labels": json.dumps(["people"]),
        "descriptions": json.dumps(["find all manequins dressed as construction workers"]),
    }

    response = requests.post(
        "http://localhost:8000/single-detect?model_name=gemini-2.5-flash-preview-04-17",
        headers={"X-API-Key": "tym_razem_to_musi_poleciec"},
        files=pictures,
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
            "barrell": (255, 255, 0),
            "palette": (0, 255, 255),
            "person": (255, 0, 0),
            "car": (0, 0, 255),
        }[detection["label"]]
        ### ZMIANY RAPORTOWANIE ###
        push_point.push_detection_to_firebase(detection, (y, x), temp_path)
        ### KONIEC ZMIAN ###
        cv2.rectangle(image, (x, y), (x + width, y + height), color, 8)
        cv2.putText(
            image, detection["label"], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 8
        )
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.savefig("plot" + str(i) + ".png")

############################ OUTFIT CHECK #######################################
    response = requests.post(
        "http://localhost:8000/qa",
        headers={"X-API-Key": "tym_razem_to_musi_poleciec"},
        files={"file": ("image.png", img_bytes, "image/png")},
        data={
            "question": "How many manequins are there? How many of them are weraing yellow helmets and yellow or red reflective vests?"
        },
    )
    print(response.json())

########################### GRAFITTI CHECK ############################################
    response = requests.post(
        "http://localhost:8000/qa",
        headers={"X-API-Key": "tym_razem_to_musi_poleciec"},
        files={"file": ("image.png", img_bytes, "image/png")},
        data={
            "question": "Are there yellow grafittis?"
        },
    )
    print(response.json())

    push_point.generate_points()
