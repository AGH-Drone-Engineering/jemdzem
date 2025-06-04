import requests
import matplotlib.pyplot as plt
import cv2
import os
import json
import sys
import datetime
import math

import raporting.push_point as push_point
from aruco_detection import detect_and_draw_aruco, ARUCO_DICTS


def pixel_to_gps(x, y, img_width, img_height):
    lat_top, lat_bottom = 50.272778, 50.272500
    lon_left, lon_right = 18.670833, 18.671111
    lon = lon_left + (x / img_width) * (lon_right - lon_left)
    lat = lat_top + (y / img_height) * (lat_bottom - lat_top)
    return lat, lon


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def is_far_enough(x, y, reference_points, min_distance=70):
    return all(distance((x, y), ref) >= min_distance for ref in reference_points)


def detect_and_annotate(image, img_bytes, ref_image_path, label, description, reference_points=None, color=(255, 255, 255), temp_folder=None, output_folder=None):
    pictures = [("file", ("image.png", img_bytes, "image/png"))]

    ref_image = cv2.imread(ref_image_path)
    _, ref_img_encoded = cv2.imencode(".png", ref_image)
    ref_img_bytes = ref_img_encoded.tobytes()
    pictures.append(("ref_file", (os.path.basename(ref_image_path), ref_img_bytes, "image/png")))

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

    print(response)
    if not response.ok:
        print(f"Detection for {label} failed.")
        return

    detections = response.json()
    accepted_detections = 0

    for detection in detections:
        x = int(detection["x"] * image.shape[1])
        y = int(detection["y"] * image.shape[0])
        width = int(detection["width"] * image.shape[1])
        height = int(detection["height"] * image.shape[0])

        if reference_points and not is_far_enough(x, y, reference_points):
            print(f"Skipped {label} at ({x}, {y}) - too close to reference point")
            continue

        lat, lon = pixel_to_gps(x, y, image.shape[1], image.shape[0])
        temp_path = os.path.join(temp_folder, f"detection_{label}_{x}_{y}.png")
        cropped_image = image[y:y+height, x:x+width]
        cv2.imwrite(temp_path, cropped_image)
        push_point.push_detection_to_firebase(detection, (lat, lon), temp_path)

        cv2.rectangle(image, (x, y), (x + width, y + height), color, 8)
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 8)

        accepted_detections += 1

    # Wysyłka informacji o brakujących obiektach
    if reference_points and accepted_detections < len(reference_points):
        missing_count = len(reference_points) - accepted_detections
        msg = f"Missing {missing_count} {label}(s)"
        print(msg)
        push_point.answer_missing(label, msg)


if __name__ == "__main__":
    push_point.clear_points()

    # Punkty referencyjne
    barrells = [(170, 140), (210, 580), (830, 180)]
    palettes = [(390, 370)]

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
        # ("pipe", "find orange pipe", "inspekcja/rura_urwana.JPG"),
        ("palette", "find all wooden pallettes", "inspekcja/paleta.png"),
    ]

    # Mapa punktów referencyjnych (dodaj tu więcej jeśli trzeba)
    reference_points_map = {
        "barrell": barrells,
        "palette": palettes,
        # "pipe": None  # nie trzeba wpisywać, jeśli brak
    }

    for label, description, ref_path in detections_info:
        reference_points = reference_points_map.get(label)  # None jeśli nie istnieje
        detect_and_annotate(
            image=image,
            img_bytes=img_bytes,
            ref_image_path=os.path.join(base_path, ref_path),
            label=label,
            description=description,
            reference_points=reference_points,
            color=color_map.get(label, (255, 255, 255)),
            temp_folder=temp_folder,
            output_folder=output_folder
        )

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
            "How many people or manequins are there? How many of them are wearing yellow or red helmets? How many of themare weraing yellow or red reflective vests?"
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
