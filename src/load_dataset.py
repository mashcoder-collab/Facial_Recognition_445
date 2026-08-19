import os
import cv2
import numpy as np

# Dataset path
dataset_path = "../dataset"

faces = []
labels = []

# Use only first 10 persons
for person_id in range(1,11):

    folder = f"s{person_id}"
    path = os.path.join(dataset_path, folder)

    for image_name in os.listdir(path):

        image_path = os.path.join(path, image_name)

        # Read image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        # Resize image
        img = cv2.resize(img, (50,50))

        # Convert image into a vector
        faces.append(img.flatten())

        labels.append(person_id)

faces = np.array(faces)
labels = np.array(labels)

print("Dataset Loaded Successfully")
print("Faces shape:", faces.shape)
print("Labels shape:", labels.shape)