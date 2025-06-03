"""Example script calling the ``/single-detect`` API and visualising results."""

import matplotlib.pyplot as plt
import cv2
import os
import json
import sys


if __name__ == "__main__":
    objects = ["person"]

    stream_url = 'rtsp://192.168.241.1/live'

    # Open the video stream
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("Failed to connect to the drone stream.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Display the frame
        cv2.imshow("Drone Feed", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
