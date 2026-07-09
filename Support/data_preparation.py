from sklearn.datasets import fetch_olivetti_faces
import numpy as np
import cv2
import os
import ssl
import certifi

# Fix for macOS/Homebrew Python SSL certificate verification errors when
# downloading the dataset over HTTPS.
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

OUTPUT_DIR = "data/processed"
N_SUBJECTS = 10          # assignment requires 10 individuals
IMAGES_PER_SUBJECT = 10  # Olivetti gives 10 expressions/poses per subject

def main():
    print("Downloading Olivetti Faces dataset via scikit-learn ...")
    olivetti = fetch_olivetti_faces(data_home="data/raw", shuffle=False)

    images = olivetti.images  
    targets = olivetti.target  

    print(f"Full dataset: {images.shape[0]} images, "
          f"{len(np.unique(targets))} unique subjects, "
          f"image size {images.shape[1]}x{images.shape[2]}")

    selected_subject_ids = list(range(N_SUBJECTS))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_saved = 0
    for subject_id in selected_subject_ids:
        subject_dir = os.path.join(OUTPUT_DIR, f"person_{subject_id:02d}")
        os.makedirs(subject_dir, exist_ok=True)

        idxs = np.where(targets == subject_id)[0]

        for i, idx in enumerate(idxs):
            img_float = images[idx]                         
            img_uint8 = (img_float * 255).astype(np.uint8)     
            out_path = os.path.join(subject_dir, f"img_{i:02d}.jpg")
            cv2.imwrite(out_path, img_uint8)
            total_saved += 1

    print(f"Saved {total_saved} images for {N_SUBJECTS} subjects into '{OUTPUT_DIR}/'")
    print("Folder layout example:")
    for subject_id in selected_subject_ids[:2]:
        subject_dir = os.path.join(OUTPUT_DIR, f"person_{subject_id:02d}")
        files = sorted(os.listdir(subject_dir))
        print(f"  {subject_dir}/ -> {files}")

if __name__ == "__main__":
    main()