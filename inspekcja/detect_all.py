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
    lat_top = 50.272778
    lat_bottom = 50.272500
    lon_left = 18.670833
    lon_right = 18.671111
    lon = lon_left + (x / img_width) * (lon_right - lon_left)
    lat = lat_top + (y / img_height) * (lat_bottom - lat_top)
    return lat, lon

if __name__ == "__main__":
    ### Przygotowanie folderów ###
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
    
   
##################### DETECT BARRELL/PIPE/PALETTE ###############
    objects = ["barrell"]

    descriptions = ["find all blue barrels with black lids"]

    #image = cv2.imread(os.path.join(os.path.dirname(__file__), sys.argv[1]))
    _, img_encoded = cv2.imencode(".png", image)
    img_bytes = img_encoded.tobytes()
    ### ArUco Detection ###
    detect_and_draw_aruco(image, ARUCO_DICTS, pixel_to_gps, push_point,  output_folder, temp_folder)



    ref_image = cv2.imread(os.path.join(os.path.dirname(__file__), "inspekcja/beczka.png"))
    pictures = [("file", ("image.png", img_bytes, "image/png"))]
    _, ref_img_encoded = cv2.imencode(".png", ref_image)
    ref_img_bytes = ref_img_encoded.tobytes()
    name = "ref_file"
    pictures.append((name, ("inspekcja/beczka.png", ref_img_bytes, "image/png")))

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
        # push_point.push_detection_to_firebase(detection, (y,x), temp_folder)
        ### KONIEC ZMIAN ###
        cv2.rectangle(image, (x, y), (x + width, y + height), color, 8)
        cv2.putText(
            image, detection["label"], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 8
        )

        lat, lon = pixel_to_gps(x, y, image.shape[1], image.shape[0])
        temp_path = os.path.join(temp_folder, f"detection_{detection['label']}_{x}_{y}.png")
        cropped_image = image[y:y+height, x:x+width]
        cv2.imwrite(temp_path, cropped_image)
        push_point.push_detection_to_firebase(detection, (lat, lon), temp_path)
    #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    print("Completed detection of",objects)
    cv2.imwrite(os.path.join(output_folder, "plot_barrell.png"), image)


    objects = ["pipe"]
    descriptions = ["find orange pipe"]

    #image = cv2.imread(os.path.join(os.path.dirname(__file__), sys.argv[1]))

    ref_image = cv2.imread(os.path.join(os.path.dirname(__file__), "inspekcja/rura_urwana.JPG"))
    pictures = [("file", ("image.png", img_bytes, "image/png"))]
    _, ref_img_encoded = cv2.imencode(".png", ref_image)
    ref_img_bytes = ref_img_encoded.tobytes()
    name = "ref_file"
    pictures.append((name, ("inspekcja/rura_urwana.JPG", ref_img_bytes, "image/png")))

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
        temp_path = os.path.join(temp_folder, f"detection_{detection['label']}_{x}_{y}.png")
        cropped_image = image[y:y+height, x:x+width]
        cv2.imwrite(temp_path, cropped_image)
        #push_point.push_detection_to_firebase(detection, (y,x), temp_path)
        ### KONIEC ZMIAN ###
        cv2.rectangle(image, (x, y), (x + width, y + height), color, 8)
        cv2.putText(
            image, detection["label"], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 8
        )

        lat, lon = pixel_to_gps(x, y, image.shape[1], image.shape[0])

        push_point.push_detection_to_firebase(detection, (lat, lon), temp_path)
    #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    print("Completed detection of",objects)
    cv2.imwrite(os.path.join(output_folder, "plot_pipe.png"), image)

    objects = ["palette"]
    descriptions = ["find all wooden pallettes"]

    #image = cv2.imread(os.path.join(os.path.dirname(__file__), sys.argv[1]))

    ref_image = cv2.imread(os.path.join(os.path.dirname(__file__), "inspekcja/paleta.png"))
    pictures = [("file", ("image.png", img_bytes, "image/png"))]
    _, ref_img_encoded = cv2.imencode(".png", ref_image)
    ref_img_bytes = ref_img_encoded.tobytes()
    name = "ref_file"
    pictures.append((name, ("inspekcja/paleta.png", ref_img_bytes, "image/png")))

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
        temp_path = os.path.join(temp_folder, f"detection_{detection['label']}_{x}_{y}.png")
        cropped_image = image[y:y+height, x:x+width]
        cv2.imwrite(temp_path, cropped_image)
        #push_point.push_detection_to_firebase(detection, (y,x), temp_path)
        ### KONIEC ZMIAN ###
        cv2.rectangle(image, (x, y), (x + width, y + height), color, 8)
        cv2.putText(
            image, detection["label"], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 8
        )

        lat, lon = pixel_to_gps(x, y, image.shape[1], image.shape[0])

        push_point.push_detection_to_firebase(detection, (lat, lon), temp_path)
    #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    print("Completed detection of",objects)
    cv2.imwrite(os.path.join(output_folder, "plot_palette.png"), image)



    ### Person Detection ###
    pictures = [("file", ("image.png", img_bytes, "image/png"))]

    data = {
        "labels": json.dumps(["person"]),
        "descriptions": json.dumps(["find all people and manequins"]),
    }

    response = requests.post(
        "http://localhost:8000/single-detect?model_name=gemini-2.5-flash-preview-04-17",
        headers={"X-API-Key": "tym_razem_to_musi_poleciec"},
        files=pictures,
        data=data,
    )

    person_detections = []
    if response.ok:
        detections = response.json()
        for detection in detections:
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

            color = {
                "pipe": (0, 255, 0),
                "barrell": (255, 255, 0),
                "palette": (0, 255, 255),
                "person": (255, 0, 0),
                "car": (0, 0, 255),
            }.get(detection["label"], (255, 255, 255))

            cv2.rectangle(image, (x, y), (x + width, y + height), color, 8)
            cv2.putText(image, detection["label"], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 8)
            temp_path = os.path.join(temp_folder, f"detection_{detection['label']}_{x}_{y}.png")
            cropped_image = image[y:y+height, x:x+width]
            cv2.imwrite(temp_path, cropped_image)
            push_point.push_detection_to_firebase(detection, (lat, lon), temp_path)

        with open(os.path.join(output_folder, "person_detections.json"), "w") as f:
            json.dump(person_detections, f, indent=2)

    cv2.imwrite(os.path.join(output_folder, "plot_people.png"), image)

##################DETECT CAR#################################################################

   pictures = [("file", ("image.png", img_bytes, "image/png"))]

    data = {
        "labels": json.dumps(["car"]),
        "descriptions": json.dumps(["find all cars"]),
    }

    response = requests.post(
        "http://localhost:8000/single-detect?model_name=gemini-2.5-flash-preview-04-17",
        headers={"X-API-Key": "tym_razem_to_musi_poleciec"},
        files=pictures,
        data=data,
    )

    person_detections = []
    if response.ok:
        detections = response.json()
        for detection in detections:
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

            color = {
                "pipe": (0, 255, 0),
                "barrell": (255, 255, 0),
                "palette": (0, 255, 255),
                "person": (255, 0, 0),
                "car": (0, 0, 255),
            }.get(detection["label"], (255, 255, 255))

            cv2.rectangle(image, (x, y), (x + width, y + height), color, 8)
            cv2.putText(image, detection["label"], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 8)
            temp_path = os.path.join(temp_folder, f"detection_{detection['label']}_{x}_{y}.png")
            cropped_image = image[y:y+height, x:x+width]
            cv2.imwrite(temp_path, cropped_image)
            push_point.push_detection_to_firebase(detection, (lat, lon), temp_path)

        with open(os.path.join(output_folder, "car_detections.json"), "w") as f:
            json.dump(person_detections, f, indent=2)

    cv2.imwrite(os.path.join(output_folder, "plot_cars.png"), image)

    ### QA - outfit check ###
    outfit_response = requests.post(
        "http://localhost:8000/qa",
        headers={"X-API-Key": "tym_razem_to_musi_poleciec"},
        files={"file": ("image.png", img_bytes, "image/png")},
        data={
            "question": "How many people or manequins are there? How many of them are weraing yellow or red helmets? How many of themare weraing yellow or red reflective vests?"
        },
    )
    jjson=outfit_response.json()
    push_point.answer(jjson["answer"])
    with open(os.path.join(output_folder, "qa_outfit.json"), "w") as f:
        json.dump(jjson, f, indent=2)

    ### QA - graffiti check ###
    graffiti_response = requests.post(
        "http://localhost:8000/qa",
        headers={"X-API-Key": "tym_razem_to_musi_poleciec"},
        files={"file": ("image.png", img_bytes, "image/png")},
        data={
            "question": "Are there yellow grafittis?"
        },
    )
    with open(os.path.join(output_folder, "qa_graffiti.json"), "w") as f:
        json.dump(graffiti_response.json(), f, indent=2)

    ### Save firebase points ###
    push_point.generate_points()

    print(f"\n Mission complete. Results saved in: {output_folder}")
