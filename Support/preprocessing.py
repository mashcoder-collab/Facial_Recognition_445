import os
import cv2
import numpy as np
import ssl
import certifi

# Fix for macOS/Homebrew Python SSL certificate verification errors when
# downloading the dataset over HTTPS.
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

PROCESSED_DIR = "data/processed"
TARGET_SIZE = (64, 64)

# OpenCV ships pretrained Haar Cascade XML files inside its install path.
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(CASCADE_PATH)


def detect_and_crop_face(gray_img):
    faces = face_detector.detectMultiScale(
        gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_crop = gray_img[y:y + h, x:x + w]
    else:
        face_crop = gray_img 

    face_resized = cv2.resize(face_crop, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    face_equalized = cv2.equalizeHist(face_resized)
    return face_equalized


def build_dataset():
    X, y, filepaths = [], [], []
    detected_count, fallback_count = 0, 0

    person_dirs = sorted(
        d for d in os.listdir(PROCESSED_DIR)
        if os.path.isdir(os.path.join(PROCESSED_DIR, d))
    )

    for label_idx, person_dir in enumerate(person_dirs):
        person_path = os.path.join(PROCESSED_DIR, person_dir)
        image_files = sorted(os.listdir(person_path))

        for image_file in image_files:
            img_path = os.path.join(person_path, image_file)
            gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            faces = face_detector.detectMultiScale(
                gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            if len(faces) > 0:
                detected_count += 1
            else:
                fallback_count += 1

            processed_face = detect_and_crop_face(gray_img)

            normalized = processed_face.astype(np.float32) / 255.0
            X.append(normalized.flatten())
            y.append(label_idx)
            filepaths.append(img_path)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print(f"Built feature matrix X: {X.shape}  (n_samples x n_pixels)")
    print(f"Labels y: {y.shape}, classes: {np.unique(y)}")
    print(f"Haar cascade detected a face directly in {detected_count}/{len(y)} images "
          f"({fallback_count} used full-image fallback)")

    os.makedirs("data/processed_arrays", exist_ok=True)
    np.save("data/processed_arrays/X.npy", X)
    np.save("data/processed_arrays/y.npy", y)

    label_map = {i: name for i, name in enumerate(person_dirs)}
    import json
    with open("data/processed_arrays/label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    return X, y


if __name__ == "__main__":
    build_dataset()
