"""Collect webcam samples of the dataset portraits shown on a phone.

For each phone_test_images/person_N.jpg: press N, then C.  The program
captures 20 frames of the largest detected face.  Press Q when all ten
people have been collected, then run train_face_recognizer.py.
"""

from pathlib import Path

import cv2
import numpy as np

from train_face_recognizer import FACE_SIZE, preprocess_face


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_PATH = PROJECT_ROOT / "models" / "phone_templates.npz"
SAMPLES_PER_PERSON = 20


def main() -> None:
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    camera = cv2.VideoCapture(0)
    if cascade.empty() or not camera.isOpened():
        raise RuntimeError("Could not start face detection or webcam.")

    features: list[np.ndarray] = []
    labels: list[int] = []
    active_person = 1
    remaining = 0
    frame_gap = 0
    print("Show person_N.jpg on the phone. Press N, then C to collect 20 samples.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.15, minNeighbors=7, minSize=(80, 80)
            )
            largest = max(faces, key=lambda box: box[2] * box[3]) if len(faces) else None
            if largest is not None:
                x, y, w, h = largest
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)
                if remaining and frame_gap == 0:
                    features.append(preprocess_face(gray[y : y + h, x : x + w]))
                    labels.append(active_person)
                    remaining -= 1
                    frame_gap = 2
            frame_gap = max(0, frame_gap - 1)
            status = f"Person {active_person} | samples remaining: {remaining}"
            cv2.putText(frame, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, "1-9/0 select | C capture | Q save and quit", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.imshow("Phone calibration", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if ord("1") <= key <= ord("9"):
                active_person = key - ord("0")
            elif key == ord("0"):
                active_person = 10
            elif key in (ord("c"), ord("C")):
                remaining = SAMPLES_PER_PERSON
                frame_gap = 0
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if not features:
        print("No samples collected; nothing saved.")
        return
    TEMPLATES_PATH.parent.mkdir(exist_ok=True)
    new_features = np.asarray(features)
    new_labels = np.asarray(labels)
    if TEMPLATES_PATH.exists():
        existing = np.load(TEMPLATES_PATH)
        new_features = np.vstack((existing["features"], new_features))
        new_labels = np.concatenate((existing["labels"], new_labels))
    np.savez_compressed(TEMPLATES_PATH, features=new_features, labels=new_labels)
    print(f"Saved {len(labels)} new samples; total is now {len(new_labels)}.")
    print("Now run: python src/train_face_recognizer.py")


if __name__ == "__main__":
    main()
