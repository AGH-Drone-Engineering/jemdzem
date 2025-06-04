import requests
import matplotlib.pyplot as plt
import cv2
import os
import json
import sys
import datetime

import raporting.push_point as push_point
from aruco_detection import detect_and_draw_aruco, ARUCO_DICTS


def pixel_to_gps(x, y, img_width, img_height):
    lat_top, lat_bottom = 50.272778, 50.272500
    lon_left, lon_right = 18.670833, 18.671111
    lon = lon_left + (x / img_width) * (lon_right - lon_left)
    lat = lat_top + (y / img_height) * (lat_bottom - lat_top)
    return lat, lon


def detect_and_annotate(image, label, description, ref_image_path, img_bytes, temp_folder, output_folder, color_map):
    print(f"Starting detection of {label}")
    
    ref_image = cv2.imread(ref_image_path)
    _, ref_img_encoded = cv2.imencode(".png", ref_image)
    ref_img_bytes = ref_img_encoded.tobytes()

    pictures = [
        ("file", ("image.png", img_bytes, "image/png")),
        ("ref_file", (os.path.basename(ref_image_path), ref_img_bytes, "image/png")),
    ]

    data = {
        "labels": json.dumps([label]),
        "descriptions": json.dumps([description]),
    }

    response = requests.post(
        "http://localhost:8000/single-detect?model_name=gemini-2.5-flash-preview-04-17",
        headers={"X-API-Key": "tym_razem_to_musi_poleciec"},
        files=pictures,
        data=data,
    )

    if not response.ok:
        print(f"Detection failed for {label}: {response.status_code}")
        return

    detections = response.json()
    for detection in detections:
        x = int(detection["x"] * image.shape[1])
        y = int(detection["y"] * image.shape[0])
        width = int(detection["width"] * image.shape[1])
        height = int(detection["height"] * image.shape[0])
        lat, lon = pixel_to_gps(x, y, image.shape[1], image.shape[0])
        color = color_map.get(detection["label"], (255, 255, 255))

        # Save crop and push detection
        temp_path = os.path.join(temp_folder, f"detection_{detection['label']}_{x}_{y}.png")
        cropped_image = image[y:y+height, x:x+width]
        cv2.imwrite(temp_path, cropped_image)
        push_point.push_detection_to_firebase(detection, (lat, lon), temp_path)

        # Draw rectangle
        cv2.rectangle(image, (x, y), (x + width, y + height), color, 8)
        cv2.putText(image, detection["label"], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 8)

    print(f"Completed detection of {label}")
    output_path = os.path.join(output_folder, f"plot_{label}.png")
    cv2.imwrite(output_path, image)


if __name__ == "__main__":
    push_point.clear_points()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_path = os.path.dirname(__file__)
    temp_folder = os.path.join(base_path, "temp_detections")
    output_folder = os.path.join(base_path, "mission_logs", timestamp)
    os.makedirs(temp_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    stream_url = 'rtsp://192.168.241.1/live'
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("Failed to connect to the drone stream.")
        exit()

    ret, image = cap.read()
    if not ret:
        print("Failed to grab frame.")
        exit()

    print("Connected to drone stream")

    detect_and_draw_aruco(image, ARUCO_DICTS, pixel_to_gps, push_point, output_folder, temp_folder)

    _, img_encoded = cv2.imencode(".png", image)
    img_bytes = img_encoded.tobytes()

    color_map = {
        "pipe": (0, 255, 0),
        "barrell": (255, 255, 0),
        "palette": (0, 255, 255),
        "person": (255, 0, 0),
        "car": (0, 0, 255),
    }

    detections_info = [
        ("barrell", "find all blue barrels with black lids", "inspekcja/beczka.png"),
        ("pipe", "find orange pipe", "inspekcja/rura_urwana.JPG"),
        ("palette", "find all wooden pallettes", "inspekcja/paleta.png"),
    ]

    for label, description, ref_path in detections_info:
        detect_and_annotate(image, label, description, os.path.join(base_path, ref_path), img_bytes, temp_folder, output_folder, color_map)

    # Person detection
    person_data = {
        "labels": json.dumps(["person"]),
        "descriptions": json.dumps(["find all people and manequins"]),
    }
    response = requests.post(
        "http://localhost:8000/single-detect?model_name=gemini-2.5-flash-preview-04-17",
        headers={"X-API-Key": "tym_razem_to_musi_poleciec"},
        files=[("file", ("image.png", img_bytes, "image/png"))],
        data=person_data,
    )

    person_detections = []
    if response.ok:
        for detection in response.json():
            x = int(detection["x"] * image.shape[1])
            y = int(detection["y"] * image.shape[0])
            width = int(detection["width"] * image.shape[1])
            height = int(detection["height"] * image.shape[0])
            lat, lon = pixel_to_gps(x, y, image.shape[1], image.shape[0])

            person_detections.append({
                "label": detection["label"],
                "bbox_px": [x, y, width, height],
                "gps": {"lat": lat, "lon": lon}
            })

            color = color_map.get(detection["label"], (255, 255, 255))
            cv2.rectangle(image, (x, y), (x + width, y + height), color, 8)
            cv2.putText(image, detection["label"], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 8)

            temp_path = os.path.join(temp_folder, f"detection_{detection['label']}_{x}_{y}.png")
            cropped_image = image[y:y+height, x:x+width]
            cv2.imwrite(temp_path, cropped_image)
            push_point.push_detection_to_firebase(detection, (lat, lon), temp_path)

        with open(os.path.join(output_folder, "person_detections.json"), "w") as f:
            json.dump(person_detections, f, indent=2)

    cv2.imwrite(os.path.join(output_folder, "plot_people.png"), image)

    # QA Outfit
    qa_questions = [
        (
            "qa_outfit.json",
            "How many people or manequins are there? How many of them are weraing yellow or red helmets? How many of themare weraing yellow or red reflective vests?"
        ),
        (
            "qa_graffiti.json",
            "Are there yellow grafittis?"
        )
    ]

    for filename, question in qa_questions:
        qa_response = requests.post(
            "http://localhost:8000/qa",
            headers={"X-API-Key": "tym_razem_to_musi_poleciec"},
            files={"file": ("image.png", img_bytes, "image/png")},
            data={"question": question},
        )
        result = qa_response.json()
        if "answer" in result:
            push_point.answer(result["answer"])
        with open(os.path.join(output_folder, filename), "w") as f:
            json.dump(result, f, indent=2)

    push_point.generate_points()

    print(f"\nMission complete. Results saved in: {output_folder}")
