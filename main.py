import os
import ssl
import json
import time

import certifi
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

RAW_DIR = "Data/raw"
PROCESSED_DIR = "Data/processed"
ARRAYS_DIR = "Data/processed_arrays"
MODELS_DIR = "models"
FIGURES_DIR = "outputs/figures"
OUTPUTS_DIR = "outputs"

N_SUBJECTS = 10          # assignment requires 10 individuals
IMG_SIZE = (64, 64)
VARIANCE_TARGET = 0.95   # PCA keeps enough components for 95% of variance
TEST_SIZE = 0.30         # 70/30 stratified train/test split
CONFIDENCE_THRESHOLD = 0.55

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(CASCADE_PATH)


def banner(title):
    print("\n" + "=" * 78)
    print(f" {title}")
    print("=" * 78)


def step1_prepare_dataset():
    banner("STEP 1/7 -- Dataset Preparation")

    # Fix for a common macOS/Homebrew Python SSL error when downloading the
    # dataset over HTTPS (harmless on systems where it isn't needed).
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

    print("Downloading Olivetti Faces dataset via scikit-learn (cached after first run)...")
    olivetti = fetch_olivetti_faces(data_home=RAW_DIR, shuffle=False)
    images, targets = olivetti.images, olivetti.target

    print(f"Full dataset: {images.shape[0]} images, {len(np.unique(targets))} subjects, "
          f"{images.shape[1]}x{images.shape[2]} pixels each")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    total_saved = 0
    for subject_id in range(N_SUBJECTS):
        subject_dir = os.path.join(PROCESSED_DIR, f"person_{subject_id:02d}")
        os.makedirs(subject_dir, exist_ok=True)
        idxs = np.where(targets == subject_id)[0]
        for i, idx in enumerate(idxs):
            img_uint8 = (images[idx] * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(subject_dir, f"img_{i:02d}.jpg"), img_uint8)
            total_saved += 1

    print(f"Saved {total_saved} images across {N_SUBJECTS} subjects into '{PROCESSED_DIR}/'")

def _detect_and_crop_face(gray_img):
    faces = face_detector.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_crop = gray_img[y:y + h, x:x + w]
    else:
        face_crop = gray_img  # fallback: use full image
    face_resized = cv2.resize(face_crop, IMG_SIZE, interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(face_resized)


def step2_preprocess():
    banner("STEP 2/7 -- Preprocessing (OpenCV)")

    X, y, label_map = [], [], {}
    person_dirs = sorted(d for d in os.listdir(PROCESSED_DIR)
                          if os.path.isdir(os.path.join(PROCESSED_DIR, d)))

    for label_idx, person_dir in enumerate(person_dirs):
        label_map[label_idx] = person_dir
        person_path = os.path.join(PROCESSED_DIR, person_dir)
        for image_file in sorted(os.listdir(person_path)):
            gray_img = cv2.imread(os.path.join(person_path, image_file), cv2.IMREAD_GRAYSCALE)
            processed = _detect_and_crop_face(gray_img)
            X.append((processed.astype(np.float32) / 255.0).flatten())
            y.append(label_idx)

    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)
    print(f"Built feature matrix X: {X.shape}  |  labels y: {y.shape}")

    os.makedirs(ARRAYS_DIR, exist_ok=True)
    np.save(os.path.join(ARRAYS_DIR, "X.npy"), X)
    np.save(os.path.join(ARRAYS_DIR, "y.npy"), y)
    with open(os.path.join(ARRAYS_DIR, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    return X, y

def step3_split(X, y):
    banner("STEP 3/7 -- Stratified Train/Test Split")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=42
    )
    print(f"Train: {X_train.shape[0]} images | Test: {X_test.shape[0]} images")
    print(f"Train class distribution: {np.bincount(y_train)}")
    print(f"Test class distribution:  {np.bincount(y_test)}")

    for name, arr in [("X_train", X_train), ("X_test", X_test),
                       ("y_train", y_train), ("y_test", y_test)]:
        np.save(os.path.join(ARRAYS_DIR, f"{name}.npy"), arr)

    return X_train, X_test, y_train, y_test

def step4_pca(X_train, X_test):
    banner("STEP 4/7 -- Feature Extraction (PCA / Eigenfaces)")

    pca = PCA(n_components=VARIANCE_TARGET, svd_solver="full", whiten=True, random_state=42)
    pca.fit(X_train)
    n_components = pca.n_components_
    print(f"PCA kept {n_components} components for {VARIANCE_TARGET*100:.0f}% variance "
          f"({X_train.shape[1]} pixels -> {n_components} features, "
          f"{X_train.shape[1]/n_components:.1f}x reduction)")

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(pca, os.path.join(MODELS_DIR, "pca_model.pkl"))

    # Visualize mean face + top eigenfaces
    os.makedirs(FIGURES_DIR, exist_ok=True)
    mean_face = pca.mean_.reshape(IMG_SIZE)
    eigenfaces = pca.components_.reshape((n_components, *IMG_SIZE))

    fig, axes = plt.subplots(2, 6, figsize=(15, 5))
    flat_axes = axes.flatten()
    flat_axes[0].imshow(mean_face, cmap="gray"); flat_axes[0].set_title("Mean Face"); flat_axes[0].axis("off")
    for i in range(min(10, n_components)):
        flat_axes[i + 1].imshow(eigenfaces[i], cmap="gray")
        flat_axes[i + 1].set_title(f"Eigenface {i+1}")
        flat_axes[i + 1].axis("off")
    for j in range(min(10, n_components) + 1, len(flat_axes)):
        flat_axes[j].axis("off")
    plt.suptitle("Mean Face and Top Eigenfaces")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "eigenfaces.png"), dpi=130)
    plt.close()
    print(f"Saved {FIGURES_DIR}/eigenfaces.png")

    return pca, X_train_pca, X_test_pca

def step5_train_models(X_train_pca, X_test_pca, y_train, y_test):
    banner("STEP 5/7 -- Model Training & Comparison")

    results, trained = {}, {}

    print("Training SVM (RBF) with GridSearchCV...")
    svm_grid = GridSearchCV(
        SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42),
        {"C": [1, 10, 50, 100, 500, 1000], "gamma": [0.0001, 0.001, 0.01, 0.1]},
        cv=3, n_jobs=-1,
    )
    svm_grid.fit(X_train_pca, y_train)
    results["SVM (RBF)"] = accuracy_score(y_test, svm_grid.predict(X_test_pca))
    trained["SVM (RBF)"] = svm_grid.best_estimator_
    print(f"  Best params: {svm_grid.best_params_} | Test accuracy: {results['SVM (RBF)']:.4f}")

    print("Training KNN...")
    knn_grid = GridSearchCV(
        KNeighborsClassifier(),
        {"n_neighbors": [1, 3, 5], "weights": ["uniform", "distance"]},
        cv=3, n_jobs=-1,
    )
    knn_grid.fit(X_train_pca, y_train)
    results["KNN"] = accuracy_score(y_test, knn_grid.predict(X_test_pca))
    trained["KNN"] = knn_grid.best_estimator_
    print(f"  Best params: {knn_grid.best_params_} | Test accuracy: {results['KNN']:.4f}")

    print("Training Logistic Regression...")
    logreg = LogisticRegression(max_iter=5000)
    logreg.fit(X_train_pca, y_train)
    results["Logistic Regression"] = accuracy_score(y_test, logreg.predict(X_test_pca))
    trained["Logistic Regression"] = logreg
    print(f"  Test accuracy: {results['Logistic Regression']:.4f}")

    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_train_pca, y_train)
    results["Random Forest"] = accuracy_score(y_test, rf.predict(X_test_pca))
    trained["Random Forest"] = rf
    print(f"  Test accuracy: {results['Random Forest']:.4f}")

    best_name = max(results, key=results.get)
    best_model = trained[best_name]
    print(f"\nBest model: {best_name} (test accuracy = {results[best_name]:.4f})")

    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    with open(os.path.join(MODELS_DIR, "best_model_name.txt"), "w") as f:
        f.write(best_name)
    with open(os.path.join(OUTPUTS_DIR, "model_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nFull comparison:")
    for name, acc in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"  {name:22s}: {acc:.4f}")

    return best_model, best_name, results


def step6_evaluate(best_model, best_name, results, X_test_raw, X_test_pca, y_test):
    banner("STEP 6/7 -- Evaluation")

    y_pred = best_model.predict(X_test_pca)

    # Model comparison bar chart
    plt.figure(figsize=(7, 4.5))
    names = list(results.keys())
    accs = [results[n] * 100 for n in names]
    colors = ["#2E86AB" if n != best_name else "#2ECC71" for n in names]
    bars = plt.bar(names, accs, color=colors)
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2, acc + 1, f"{acc:.1f}%", ha="center", fontweight="bold")
    plt.ylabel("Test Accuracy (%)"); plt.ylim(0, 110); plt.xticks(rotation=15)
    plt.title("Classifier Comparison on Eigenface Features")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "model_comparison.png"), dpi=130)
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[f"P{i}" for i in range(N_SUBJECTS)],
                yticklabels=[f"P{i}" for i in range(N_SUBJECTS)])
    plt.xlabel("Predicted Person"); plt.ylabel("True Person")
    plt.title(f"Confusion Matrix -- {best_name}\nOverall Accuracy: {results[best_name]*100:.2f}%")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrix.png"), dpi=130)
    plt.close()

    # Classification report
    report = classification_report(y_test, y_pred, target_names=[f"Person {i}" for i in range(N_SUBJECTS)])
    with open(os.path.join(OUTPUTS_DIR, "classification_report.txt"), "w") as f:
        f.write(f"Best model: {best_name}\n\n{report}")
    print(report)

    # Sample predictions grid
    n_show = min(15, len(y_test))
    idxs = np.random.RandomState(1).choice(len(y_test), size=n_show, replace=False)
    fig, axes = plt.subplots(3, 5, figsize=(13, 8))
    for ax, idx in zip(axes.flatten(), idxs):
        img = X_test_raw[idx].reshape(IMG_SIZE)
        correct = y_test[idx] == y_pred[idx]
        ax.imshow(img, cmap="gray")
        ax.set_title(f"True: P{y_test[idx]}  Pred: P{y_pred[idx]}",
                     color="green" if correct else "red", fontsize=10)
        ax.axis("off")
    for ax in axes.flatten()[n_show:]:
        ax.axis("off")
    plt.suptitle("Sample Test Predictions (green = correct, red = wrong)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "sample_predictions.png"), dpi=130)
    plt.close()

    print(f"\nSaved figures to {FIGURES_DIR}/: model_comparison.png, confusion_matrix.png, sample_predictions.png")
    return y_pred

def predict_identity(image_bgr_or_gray, pca, model):
    """Run the full pipeline on a single new image and return a prediction."""
    gray = cv2.cvtColor(image_bgr_or_gray, cv2.COLOR_BGR2GRAY) \
        if len(image_bgr_or_gray.shape) == 3 else image_bgr_or_gray
    processed = _detect_and_crop_face(gray)
    features = (processed.astype(np.float32) / 255.0).flatten().reshape(1, -1)
    features_pca = pca.transform(features)

    proba = model.predict_proba(features_pca)[0]
    pred_idx = int(np.argmax(proba))
    confidence = float(proba[pred_idx])
    return {
        "person_id": pred_idx,
        "confidence": round(confidence, 4),
        "accepted": confidence >= CONFIDENCE_THRESHOLD,
    }


def step7_demo(pca, best_model, label_map):
    banner("STEP 7/7 -- Live Demo (random test images)")

    demo_people = np.random.RandomState(7).choice(N_SUBJECTS, size=5, replace=False)
    print("Running the trained model on a few fresh images it did NOT train on:\n")
    print(f"{'True Person':<15}{'Predicted':<15}{'Confidence':<15}{'Accepted?':<12}{'Result'}")
    print("-" * 72)

    correct_count = 0
    for subject_id in demo_people:
        subject_dir = os.path.join(PROCESSED_DIR, label_map[str(subject_id)] if str(subject_id) in label_map
                                    else f"person_{subject_id:02d}")
        # use the last image (img_09) of that subject as a "new" test image
        img_path = os.path.join(subject_dir, "img_09.jpg")
        img = cv2.imread(img_path)
        result = predict_identity(img, pca, best_model)

        is_correct = result["person_id"] == subject_id
        correct_count += is_correct
        print(f"Person {subject_id:<8}Person {result['person_id']:<8}"
              f"{result['confidence']*100:>6.1f}%       "
              f"{'YES' if result['accepted'] else 'NO':<12}{'CORRECT' if is_correct else 'WRONG'}")

    print("-" * 72)
    print(f"Demo result: {correct_count}/{len(demo_people)} identities correctly recognized")
    print(f"(Accepted = confidence cleared the {CONFIDENCE_THRESHOLD*100:.0f}% threshold used for real decisions "
          f"like login/payment approval; a low-confidence correct guess still shows NO here on purpose.)\n")

def main():
    start = time.time()
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    step1_prepare_dataset()
    X, y = step2_preprocess()
    X_train, X_test, y_train, y_test = step3_split(X, y)
    pca, X_train_pca, X_test_pca = step4_pca(X_train, X_test)
    best_model, best_name, results = step5_train_models(X_train_pca, X_test_pca, y_train, y_test)
    step6_evaluate(best_model, best_name, results, X_test, X_test_pca, y_test)

    with open(os.path.join(ARRAYS_DIR, "label_map.json")) as f:
        label_map = json.load(f)
    step7_demo(pca, best_model, label_map)

    elapsed = time.time() - start
    banner("PIPELINE COMPLETE")
    print(f"Best model: {best_name}")
    print(f"Test accuracy: {results[best_name]*100:.2f}%")
    print(f"Total runtime: {elapsed:.1f} seconds")
    print(f"\nAll figures saved in:  {FIGURES_DIR}/")
    print(f"Trained model saved in: {MODELS_DIR}/best_model.pkl")
    print(f"Classification report:  {OUTPUTS_DIR}/classification_report.txt")


if __name__ == "__main__":
    main()
