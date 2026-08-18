"""Real-time face recognition with an explicit Unknown result.

Run ``python train_face_recognizer.py`` once before using this program.
Press Q to close the camera.
"""

from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "face_recognizer.pkl"
FACE_SIZE = (50, 50)
HISTORY_LENGTH = 8
MIN_VOTES = 5
# Person 4 and 9 have more phone-screen variation.  They may use this wider
# range only when several nearby calibration templates independently agree.
SOFT_THRESHOLDS = {4: 0.62, 9: 0.62}
SOFT_MIN_NEIGHBOURS = 4


def preprocess_face(face: np.ndarray) -> np.ndarray:
    """Match the illumination-normalised representation used during training."""
    face = cv2.resize(face, FACE_SIZE, interpolation=cv2.INTER_AREA)
    face = cv2.equalizeHist(face)
    feature = face.astype(np.float32).reshape(-1)
    feature -= feature.mean()
    return feature / (np.linalg.norm(feature) + 1e-8)


def recognise(face: np.ndarray, recognizer: dict) -> tuple[int | None, float]:
    """Return a label only when it is close enough to known training faces."""
    feature = preprocess_face(face).reshape(1, -1)
    distances, indices = recognizer["knn"].kneighbors(feature, n_neighbors=5)
    distance = float(distances[0, 0])
    label = int(recognizer["labels"][indices[0, 0]])
    threshold = recognizer["thresholds"].get(label, recognizer["global_threshold"])
    neighbour_labels = recognizer["labels"][indices[0]]
    agreement = int(np.count_nonzero(neighbour_labels == label))
    is_strict_match = distance <= threshold
    is_supported_soft_match = (
        label in SOFT_THRESHOLDS
        and distance <= SOFT_THRESHOLDS[label]
        and agreement >= SOFT_MIN_NEIGHBOURS
    )
    return (label if is_strict_match or is_supported_soft_match else None), distance


def stable_label(history: deque[int | None]) -> int | None:
    """Avoid labels jumping while the detector receives noisy frames."""
    known = [label for label in history if label is not None]
    if len(known) < MIN_VOTES:
        return None
    label, votes = Counter(known).most_common(1)[0]
    return label if votes >= MIN_VOTES else None


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained recognizer found at {MODEL_PATH}. "
            "Run: python src/train_face_recognizer.py"
        )
    recognizer = joblib.load(MODEL_PATH)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if cascade.empty():
        raise RuntimeError("Could not load OpenCV's frontal-face detector.")

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions and try again.")
    histories: list[deque[int | None]] = []
    print("Webcam started. Press Q to quit.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.15, minNeighbors=7, minSize=(80, 80)
            )
            while len(histories) < len(faces):
                histories.append(deque(maxlen=HISTORY_LENGTH))
            if not len(faces):
                histories.clear()
            for index, (x, y, w, h) in enumerate(faces):
                label, distance = recognise(gray[y : y + h, x : x + w], recognizer)
                histories[index].append(label)
                confirmed = stable_label(histories[index])
                if confirmed is not None:
                    text, colour = f"Person {confirmed}", (0, 220, 0)
                elif label is not None:
                    # A valid distance has been found, but it needs a few
                    # consistent frames before becoming a final identity.
                    text, colour = f"Checking Person {label}", (0, 220, 255)
                else:
                    text, colour = "Unknown", (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
                cv2.putText(
                    frame, f"{text}  d={distance:.2f}", (x, max(25, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2,
                )
            cv2.imshow("Face Recognition (Q to quit)", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
