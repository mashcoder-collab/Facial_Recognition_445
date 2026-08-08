import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Dataset location
dataset_path = "../dataset"

faces = []
labels = []


# Load images from 10 people
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

        # Resize image
        img = cv2.resize(img, (50, 50))

        # Convert image to feature vector
        faces.append(img.flatten())

        # Assign person label
        labels.append(person_id)


faces = np.array(faces)
labels = np.array(labels)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    faces,
    labels,
    test_size=0.2,
    random_state=42
)


# Create SVM model
svm = SVC(kernel="linear")


# Train
svm.fit(X_train, y_train)


# Predict
predictions = svm.predict(X_test)


# Calculate accuracy
accuracy = accuracy_score(
    y_test,
    predictions
)

print("SVM Accuracy:", accuracy * 100, "%")