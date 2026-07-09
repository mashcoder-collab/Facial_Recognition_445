import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


dataset_path = "../dataset"

faces = []
labels = []

# Load first 10 persons
for person_id in range(1,11):

    folder = f"s{person_id}"
    path = os.path.join(dataset_path, folder)

    for image_name in os.listdir(path):

        image_path = os.path.join(path,image_name)

        img = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if img is None:
            continue

        img = cv2.resize(img,(50,50))

        faces.append(img.flatten())

        labels.append(person_id)

faces=np.array(faces)
labels=np.array(labels)


# Split data
X_train,X_test,y_train,y_test = train_test_split(
    faces,
    labels,
    test_size=0.2,
    random_state=42
)

# Create KNN model
knn=KNeighborsClassifier(n_neighbors=3)

# Train
knn.fit(X_train,y_train)

# Predict
predictions=knn.predict(X_test)

# Accuracy
accuracy=accuracy_score(
    y_test,
    predictions
)

print("KNN Accuracy:",accuracy*100,"%")