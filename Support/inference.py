import cv2
import numpy as np
import joblib

IMG_SIZE = (64, 64)
CONFIDENCE_THRESHOLD = 0.55  

pca = joblib.load("models/pca_model.pkl")
model = joblib.load("models/best_model.pkl")
with open("models/best_model_name.txt") as f:
    MODEL_NAME = f.read().strip()

cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def _preprocess(gray_img):
    faces = cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_crop = gray_img[y:y + h, x:x + w]
    else:
        face_crop = gray_img
    face_resized = cv2.resize(face_crop, IMG_SIZE, interpolation=cv2.INTER_AREA)
    face_equalized = cv2.equalizeHist(face_resized)
    normalized = face_equalized.astype(np.float32) / 255.0
    return normalized.flatten().reshape(1, -1)


def predict_identity(image_bgr_or_gray):
    """
    image_bgr_or_gray: a numpy array as returned by cv2.imread / cv2.imdecode
    Returns: dict with keys 'person_id', 'confidence', 'accepted', 'model'
    """
    if len(image_bgr_or_gray.shape) == 3:
        gray = cv2.cvtColor(image_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr_or_gray

    features = _preprocess(gray)
    features_pca = pca.transform(features)

    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(features_pca)[0]
            pred_idx = int(np.argmax(proba))
            confidence = float(proba[pred_idx])
        except Exception:
            pred_idx = int(model.predict(features_pca)[0])
            confidence = 1.0
    else:
        pred_idx = int(model.predict(features_pca)[0])
        confidence = 1.0

    accepted = confidence >= CONFIDENCE_THRESHOLD

    return {
        "person_id": pred_idx,
        "confidence": round(confidence, 4),
        "accepted": accepted,
        "model": MODEL_NAME,
    }


if __name__ == "__main__":

    sample_path = "data/processed/person_03/img_08.jpg"
    img = cv2.imread(sample_path)
    result = predict_identity(img)
    print(f"Testing on {sample_path} (true identity = person_03)")
    print(result)
