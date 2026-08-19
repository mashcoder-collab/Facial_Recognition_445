import os
import cv2
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# =====================================
# 1. Load Dataset
# =====================================

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


# =====================================
# 2. Define Models
# =====================================

models = {

    "KNN": KNeighborsClassifier(
        n_neighbors=3
    ),

    "SVM": SVC(
        kernel="linear"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


# =====================================
# 3. 5-Fold Stratified Cross-Validation
# =====================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# =====================================
# 4. Evaluate Each Model
# =====================================

for name, model in models.items():

    print("\n==============================")
    print(name)
    print("==============================")

    scores = cross_val_score(
        model,
        faces,
        labels,
        cv=cv,
        scoring="accuracy"
    )

    print("Fold Accuracies:")

    for i, score in enumerate(scores, start=1):
        print(
            f"Fold {i}: {score * 100:.2f}%"
        )

    print(
        f"Mean Accuracy: {scores.mean() * 100:.2f}%"
    )

    print(
        f"Standard Deviation: {scores.std() * 100:.2f}%"
    )