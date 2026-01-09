"""
See README.md for running instructions, examples and authors.
"""

import logging
from dataclasses import dataclass

import cv2 as cv
import numpy as np

log = logging.getLogger(__name__)

ESC = 27

# BGR colors
RED = (0, 0, 255)
BLUE = (255, 0, 0)

type Point = tuple[int, int]


@dataclass
class Face:
    x: int
    y: int
    width: int
    height: int
    center: Point
    bounding_box: np.ndarray


@dataclass
class Eye:
    x: int
    y: int
    width: int
    height: int
    center: Point


face_cascade = cv.CascadeClassifier("data/haarcascade_frontalface_default.xml")
eyes_cascade = cv.CascadeClassifier("data/haarcascade_eye_tree_eyeglasses.xml")


def main() -> None:
    logging.basicConfig()

    camera = cv.VideoCapture(0)
    ad_video = cv.VideoCapture("data/ad.webm")

    while camera.isOpened():
        frame = read_frame(camera, "camera")
        if frame is None:
            continue

        # Simplify frame for processing
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray = cv.equalizeHist(frame_gray)

        # Find all faces in the frame
        faces = detect_faces(frame_gray)
        eye_count = 0
        for face in faces:
            frame = cv.ellipse(
                frame,
                face.center,
                axes=(face.width // 2, face.height // 2),
                angle=0,
                startAngle=0,
                endAngle=360,
                color=RED,
                thickness=4,
            )

            # Find all eyes in the face
            eyes = detect_eyes(face)
            for eye in eyes:
                radius = int(round((eye.width + eye.height) * 0.25))
                frame = cv.circle(frame, eye.center, radius, color=BLUE, thickness=2)
                eye_count += 1

        cv.imshow("Camera", cv.flip(frame, 1))

        # Only play the ad if the user is looking at it
        if eye_count >= 2:
            ad_frame = read_frame(ad_video, "ad")
            if ad_frame is not None:
                cv.imshow("Advertisement", ad_frame)

        if pressed_esc():
            break

    camera.release()
    ad_video.release()
    cv.destroyAllWindows()


def read_frame(capture: cv.VideoCapture, name: str) -> np.ndarray | None:
    success, frame = capture.read()
    if not success:
        log.warning("Failed to read frame from %s", name)
        return None

    return frame


def detect_faces(frame: np.ndarray) -> list[Face]:
    """Find all faces in the frame"""

    detected_faces = face_cascade.detectMultiScale(frame)
    faces: list[Face] = []
    for x, y, w, h in detected_faces:
        center = (x + w // 2, y + h // 2)
        bounding_box = frame[y : (y + h), x : (x + w)]
        faces.append(Face(x, y, w, h, center, bounding_box))

    return faces


def detect_eyes(face: Face) -> list[Eye]:
    """Find all eyes in a face"""

    detected_eyes = eyes_cascade.detectMultiScale(face.bounding_box)
    eyes: list[Eye] = []
    for x, y, w, h in detected_eyes:
        center = (face.x + x + w // 2, face.y + y + h // 2)
        eyes.append(Eye(x, y, w, h, center))

    return eyes


def pressed_esc() -> bool:
    return cv.waitKey(5) & 0xFF == ESC


if __name__ == "__main__":
    main()
