import os
import cv2
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ==========================================
# 1. Dataset Location
# ==========================================

dataset_path = "../dataset"

faces = []
labels = []


# ==========================================
# 2. Load Images from 10 People
# ==========================================

for person_id in range(1, 11):

    folder = f"s{person_id}"
    path = os.path.join(dataset_path, folder)

    for image_name in os.listdir(path):

        image_path = os.path.join(path, image_name)

        img = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if img is None:
            continue

        # Resize image to 50 × 50
        img = cv2.resize(img, (50, 50))

        # Convert image to 1D feature vector
        faces.append(img.flatten())

        # Assign person label
        labels.append(person_id)


faces = np.array(faces)
labels = np.array(labels)


print("Faces shape:", faces.shape)
print("Labels shape:", labels.shape)


# ==========================================
# 3. Split Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    faces,
    labels,
    test_size=0.2,
    random_state=42
)


print("Training images:", len(X_train))
print("Testing images:", len(X_test))


# ==========================================
# 4. Create Random Forest Model
# ==========================================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ==========================================
# 5. Train the Model
# ==========================================

rf.fit(X_train, y_train)


# ==========================================
# 6. Test the Model
# ==========================================

predictions = rf.predict(X_test)


accuracy = accuracy_score(
    y_test,
    predictions
)

print("Random Forest Accuracy:", accuracy * 100, "%")


# ==========================================
# 7. Save the Trained Model
# ==========================================

# Create models directory if it doesn't exist
models_path = "../models"

os.makedirs(
    models_path,
    exist_ok=True
)


# Model file location
model_file = os.path.join(
    models_path,
    "random_forest_model.pkl"
)


# Save model
joblib.dump(
    rf,
    model_file
)


print("Random Forest model saved successfully!")
print("Model location:", model_file)