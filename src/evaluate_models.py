import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ==============================
# 1. Load Dataset
# ==============================

dataset_path = "../dataset"

faces = []
labels = []


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

        # Flatten image
        faces.append(img.flatten())

        # Person label
        labels.append(person_id)


faces = np.array(faces)
labels = np.array(labels)


print("Faces shape:", faces.shape)
print("Labels shape:", labels.shape)


# ==============================
# 2. Stratified Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    faces,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)


print("Training images:", len(X_train))
print("Testing images:", len(X_test))


# ==============================
# 3. Create Models
# ==============================

models = {

    "KNN": KNeighborsClassifier(n_neighbors=3),

    "SVM": SVC(kernel="linear"),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


# ==============================
# 4. Train and Evaluate
# ==============================

for name, model in models.items():

    print("\n==============================")
    print(name)
    print("==============================")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # Print results
    print("Accuracy :", accuracy * 100, "%")
    print("Precision:", precision * 100, "%")
    print("Recall   :", recall * 100, "%")
    print("F1 Score :", f1 * 100, "%")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:")
    print(cm)