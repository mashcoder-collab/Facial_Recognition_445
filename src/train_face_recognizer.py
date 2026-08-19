"""Train the webcam recognizer and calibrate its Unknown thresholds."""

from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "dataset"
MODEL_PATH = PROJECT_ROOT / "models" / "face_recognizer.pkl"
PHONE_TEMPLATES_PATH = PROJECT_ROOT / "models" / "phone_templates.npz"
FACE_SIZE = (50, 50)
FACE_DETECTOR = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def preprocess_face(face: np.ndarray) -> np.ndarray:
    face = cv2.resize(face, FACE_SIZE, interpolation=cv2.INTER_AREA)
    face = cv2.equalizeHist(face)
    feature = face.astype(np.float32).reshape(-1)
    feature -= feature.mean()
    return feature / (np.linalg.norm(feature) + 1e-8)


def webcam_style_face(image: np.ndarray) -> np.ndarray | None:
    """Approximate the crop OpenCV receives when a dataset photo is on a phone."""
    screen = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    faces = FACE_DETECTOR.detectMultiScale(
        screen, scaleFactor=1.1, minNeighbors=4, minSize=(80, 80)
    )
    if not len(faces):
        return None
    x, y, width, height = max(faces, key=lambda box: box[2] * box[3])
    return screen[y : y + height, x : x + width]


def load_dataset(min_person: int = 1, max_person: int = 10) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    folders = sorted(
        (item for item in DATASET_PATH.glob("s*")
         if item.name[1:].isdigit() and min_person <= int(item.name[1:]) <= max_person),
        key=lambda item: int(item.name[1:]),
    )
    for folder in folders:
        if not folder.is_dir():
            continue
        label = int(folder.name[1:])
        for image_path in sorted(folder.glob("*.pgm")):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                features.append(preprocess_face(image))
                labels.append(label)
                # A phone/camera view often contains only the detector's face
                # crop, not the original ORL image boundary. Train on both.
                detected_face = webcam_style_face(image)
                if detected_face is not None:
                    features.append(preprocess_face(detected_face))
                    labels.append(label)
    if not features:
        raise RuntimeError(f"No .pgm face images found in {DATASET_PATH}")
    return np.asarray(features), np.asarray(labels)


def calibrate_thresholds(
    features: np.ndarray, labels: np.ndarray, outsiders: np.ndarray
) -> dict[int, float]:
    """Keep genuine faces, while rejecting faces from people not being enrolled."""
    thresholds: dict[int, float] = {}
    for label in np.unique(labels):
        own = features[labels == label]
        distances = np.linalg.norm(own[:, None] - own[None, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        thresholds[int(label)] = float(np.percentile(distances.min(axis=1), 95) * 1.15)

    # The ORL dataset includes people s11-s40 that are deliberately not enrolled.
    # Use them as negative validation data and cap each routed label's threshold.
    # This prevents the common closed-set-classifier failure: every stranger is
    # forced into the nearest enrolled person.
    calibrator = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
    calibrator.fit(features, labels)
    outsider_distances, outsider_indices = calibrator.kneighbors(outsiders, n_neighbors=1)
    routed_labels = labels[outsider_indices[:, 0]]
    for label in thresholds:
        negative_distances = outsider_distances[routed_labels == label, 0]
        if len(negative_distances):
            thresholds[label] = min(thresholds[label], float(negative_distances.min() * 0.99))
    return thresholds


def main() -> None:
    features, labels = load_dataset(max_person=10)
    outsider_features, _ = load_dataset(min_person=11, max_person=40)
    if PHONE_TEMPLATES_PATH.exists():
        templates = np.load(PHONE_TEMPLATES_PATH)
        phone_features = templates["features"]
        phone_labels = templates["labels"]
        features = np.vstack((features, phone_features))
        labels = np.concatenate((labels, phone_labels))
        print(f"Added {len(phone_labels)} phone-camera calibration samples.")
    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
    knn.fit(features, labels)
    thresholds = calibrate_thresholds(features, labels, outsider_features)
    model = {
        "knn": knn,
        "labels": labels,
        "thresholds": thresholds,
        "global_threshold": float(np.median(list(thresholds.values()))),
        "preprocessing": "equalize_hist + zero_mean_l2",
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Trained on {len(labels)} images for {len(thresholds)} people.")
    print(f"Saved recognizer: {MODEL_PATH}")
    print("Unknown distance thresholds:")
    for label, threshold in thresholds.items():
        print(f"  Person {label}: {threshold:.3f}")


if __name__ == "__main__":
    main()
