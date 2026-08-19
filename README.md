# Facial_Recognition_445# Face Recognition System — 10 Individuals, Multiple Expressions

A from-scratch, classical Machine Learning face-recognition pipeline built with
**Python, NumPy, Pandas, scikit-learn, and OpenCV**, using the classic
**Eigenfaces (PCA) + SVM** approach. The entire project runs from a single
file, `main.py`.

**Result: 93.3% test accuracy** recognizing 10 people across held-out test
images with varying facial expressions.

---

## Contributors

| Student ID | Name | GitHub |
|---|---|---|
| 2232167042 | Mehedi Hasan Dip | @Dip223 |
| 2321942642 | Sayef Ali Khan   | @mashcoder-collab |

CSE445 - Machine Learning, Section 7, Group 5, North South University

## Quick Start

```bash
# 1. Clone/open the repo, create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the entire pipeline
python main.py
```

That's it. On first run, it downloads the dataset, prepares it, trains the
model, evaluates it, and runs a live demo — all in one go, in a few seconds.
Every later run reuses the cached dataset, so it's just as fast.

---

## What `main.py` Actually Does

Running `python main.py` executes 7 steps in sequence, printing progress for
each one:

| Step | What Happens | Output |
|---|---|---|
| 1. Dataset Preparation | Downloads the Olivetti Faces dataset via scikit-learn, selects 10 subjects, saves each as real `.jpg` files organized into per-person folders | `data/processed/person_00/` … `person_09/` |
| 2. Preprocessing | OpenCV face detection → crop → resize (64×64) → histogram equalization → normalize → flatten | `data/processed_arrays/X.npy`, `y.npy` |
| 3. Train/Test Split | Stratified 70/30 split (7 train / 3 test images per person) | `X_train.npy`, `X_test.npy`, etc. |
| 4. Feature Extraction | PCA (Eigenfaces): compresses 4,096 pixels → ~42 components (95% variance retained) | `models/pca_model.pkl`, `outputs/figures/eigenfaces.png` |
| 5. Model Training | Trains & compares SVM, KNN, Logistic Regression, Random Forest; keeps the best | `models/best_model.pkl` |
| 6. Evaluation | Confusion matrix, classification report, sample prediction grid | `outputs/figures/*.png`, `outputs/classification_report.txt` |
| 7. Live Demo | Runs the trained model on 5 fresh test images right in the terminal, showing prediction + confidence + accept/reject decision | printed to console |

**Expected terminal output (abridged):**
```
================================================================
 STEP 1/7 -- Dataset Preparation
================================================================
Downloading Olivetti Faces dataset via scikit-learn (cached after first run)...
Full dataset: 400 images, 40 subjects, 64x64 pixels each
Saved 100 images across 10 subjects into 'data/processed/'
...
================================================================
 STEP 5/7 -- Model Training & Comparison
================================================================
Training SVM (RBF) with GridSearchCV...
  Best params: {'C': 10, 'gamma': 0.01} | Test accuracy: 0.9333
...
Best model: SVM (RBF) (test accuracy = 0.9333)
...
================================================================
 STEP 7/7 -- Live Demo (random test images)
================================================================
True Person    Predicted      Confidence     Accepted?   Result
------------------------------------------------------------------------
Person 8       Person 8         70.3%       YES         CORRECT
...
================================================================
 PIPELINE COMPLETE
================================================================
Best model: SVM (RBF)
Test accuracy: 93.33%
```

**Total runtime:** a few seconds after the first run (the dataset download is
the only slow part, and it's cached locally afterward).

---

## Project Outline — What We Need, and Why

| # | Component | Tool | Why we need it |
|---|-----------|------|-----------------|
| 1 | Data | Olivetti/AT&T Faces dataset | A real, well-established, ethically-sourced public face dataset — 40 people × 10 images each with varying expression; we use 10 of the 40 people as required |
| 2 | Data handling | NumPy, Pandas | NumPy gives fast array math for image matrices; Pandas-style tabular thinking underlies the metadata/label bookkeeping |
| 3 | Image I/O & preprocessing | OpenCV (cv2) | Reading images, face **detection** (Haar Cascade), resizing, histogram equalization |
| 4 | Dimensionality reduction | scikit-learn `PCA` | Compresses 4,096-pixel vectors to ~42 meaningful "Eigenface" features, avoiding overfitting on a small dataset |
| 5 | Classification | scikit-learn `SVC`, `KNeighborsClassifier`, `LogisticRegression`, `RandomForestClassifier` | The actual "who is this person" decision, compared across 4 algorithms |
| 6 | Evaluation | scikit-learn metrics, Matplotlib, Seaborn | Confusion matrix, per-class precision/recall, visual proof the model works |

### The pipeline, end to end

```
Raw Images (Olivetti dataset, 10 people x 10 images)
        │
        ▼
[1] Data Preparation  ─── download → organize into per-person folders
        │
        ▼
[2] Preprocessing (OpenCV) ─── face detection → crop → resize 64x64 →
        │                       histogram equalization → normalize [0,1]
        ▼
[3] Train/Test Split (stratified, 70/30 per person)
        │
        ▼
[4] Feature Extraction (PCA / Eigenfaces) ─── 4096 pixels → ~42 features
        │
        ▼
[5] Train Classifiers (SVM, KNN, LogReg, RandomForest) ─── pick best
        │
        ▼
[6] Evaluate (accuracy, confusion matrix, classification report)
        │
        ▼
[7] Live Demo ─── run the trained model on fresh images, print results
```

### Why classical ML (Eigenfaces + SVM) and not a deep CNN?

- This uses **NumPy / Pandas / scikit-learn / OpenCV** — the classical
  computer-vision approach to face recognition that predates and underlies
  deep learning methods, rather than a deep learning framework.
- With only 10 images per person, a deep CNN trained from scratch would
  badly overfit. PCA + SVM is specifically well-suited to small datasets.
- It directly builds on "Unsupervised Learning / PCA" and "Support Vector
  Machines" — the classical ML fundamentals this project is meant to
  demonstrate.

---

## The Dataset

**Name:** Olivetti Faces Dataset (a.k.a. **AT&T Database of Faces / ORL Database**)
**Original source:** AT&T Laboratories Cambridge, 1992–1994
**Loaded via:** `sklearn.datasets.fetch_olivetti_faces()`
**Reference:** https://scikit-learn.org/stable/datasets/real_world.html#olivetti-faces-dataset

**Why this dataset:**
- It's the textbook dataset for face recognition — the same dataset
  scikit-learn ships as a built-in loader, and the one used in decades of
  Eigenfaces literature.
- It matches the assignment brief exactly: each of the 40 people has 10
  images taken at different times, **with varying facial expressions**
  (open/closed eyes, smiling/not smiling) and minor lighting/detail changes
  (glasses/no glasses).
- Small, clean, license-friendly for a course project (attribute to AT&T
  Laboratories Cambridge, as required).

**What's used:** the first **10 of the 40 people** (as required by the
assignment brief: "10 individuals"), i.e. **100 images total** (10 people ×
10 expressions).

**Dataset stats:**
- 100 grayscale images, 64×64 pixels each
- 10 identities, 10 images per identity
- Saved as real `.jpg` files under `data/processed/person_XX/`, one folder
  per identity — mirroring how a real-world face dataset (or your own
  photos) would be organized.

> **A note on the SSL fix in `main.py`:** on some systems (particularly
> Homebrew-installed Python on macOS), the HTTPS download inside
> `fetch_olivetti_faces()` can fail with a certificate verification error.
> `main.py` includes a small fix at the start of Step 1 using the `certifi`
> package to point Python at a trusted certificate bundle — this is why
> `certifi` is in `requirements.txt`.

---

## Step-by-Step Explanation

### Step 1 — Dataset Preparation
Downloads the Olivetti dataset and saves the 10 selected identities as real
`.jpg` files in per-person folders. This sets up every later step to use
genuine OpenCV image I/O instead of only working with in-memory arrays.

### Step 2 — Preprocessing (OpenCV)
For every image: **Haar Cascade** face detection → crop to the face → resize
to 64×64 → **histogram equalization** (normalizes lighting) → scale pixels
to [0, 1] → flatten to a 4,096-length vector. All 100 vectors are stacked
into a feature matrix `X` (100 × 4096) with labels `y`.

*Engineering note:* the Olivetti images are already tightly cropped with no
background margin, so the Haar cascade mostly falls back to using the full
image — which is fine, since the image already *is* the face region. This
same detection logic is what would actively crop a face out of any new,
uncropped photograph (e.g. a webcam frame), which is why it's kept in the
pipeline rather than skipped.

### Step 3 — Train/Test Split
Stratified 70/30 split (7 train / 3 test images per person) using
`train_test_split(..., stratify=y)`, so every one of the 10 identities is
represented proportionally in both sets — essential with only 10 images per
class.

### Step 4 — Feature Extraction (PCA / Eigenfaces)
Fits **PCA** on the training set only (no data leakage), keeping enough
components to explain 95% of variance — this selects **42 components**, a
**97.5× reduction** from 4,096 raw pixels. Produces the classic "Eigenfaces"
visualization (mean face + top eigenfaces).

### Step 5 — Model Training and Comparison
Trains and compares 4 classifiers on the PCA features:

| Model | Test Accuracy |
|---|---|
| **SVM (RBF kernel, GridSearchCV-tuned)** | **93.3%** ✅ best |
| Logistic Regression | 93.3% |
| Random Forest | 90.0% |
| KNN | 60.0% |

SVM is selected as the final model (tied with Logistic Regression, chosen
since it's the historically standard pairing with Eigenfaces and supports
probability-based confidence scoring).

### Step 6 — Evaluation
Generates:
- `outputs/figures/model_comparison.png` — bar chart of all 4 models
- `outputs/figures/confusion_matrix.png` — per-person confusion matrix
- `outputs/figures/sample_predictions.png` — actual test faces with predicted vs. true label
- `outputs/classification_report.txt` — precision/recall/F1 per person

**Result: 93.3% overall accuracy**, perfect (100%) precision/recall for 8 of
10 people; the only confusion is between two specific individuals — visible
directly in the confusion matrix.

### Step 7 — Live Demo
Runs the exact same pipeline used during training (detect → crop → resize →
equalize → normalize → PCA → classify) on 5 fresh test images, printing each
prediction with a **confidence score** and whether it clears a **55%
acceptance threshold** — demonstrating that a real system built on this model
wouldn't just report "closest match," but would know when to say "not
confident enough."

---

## How to Run Everything Yourself

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

No arguments, no configuration needed — it runs start to finish. Re-running
it will re-download nothing (the dataset is cached in `data/raw/` after the
first run) and will simply regenerate all figures, the model, and the demo
output.

**To inspect results afterward:**
```bash
open outputs/figures/eigenfaces.png            # macOS
open outputs/figures/confusion_matrix.png
open outputs/figures/sample_predictions.png
open outputs/figures/model_comparison.png
cat outputs/classification_report.txt
```

---

## Project Structure

```
.
├── README.md
├── requirements.txt
├── main.py                       # <- run this
├── data/
│   ├── raw/                      # cached Olivetti dataset download
│   ├── processed/                # 10 people x 10 images, as real .jpg files
│   └── processed_arrays/         # X, y, train/test splits (.npy files)
├── models/
│   ├── pca_model.pkl
│   ├── best_model.pkl
│   └── best_model_name.txt
└── outputs/
    ├── figures/                  # eigenfaces, confusion matrix, etc.
    ├── classification_report.txt
    └── model_comparison.json
```

*(If your repo uses different folder names, e.g. `Data/`, `Support/`, just
edit the path constants — `RAW_DIR`, `PROCESSED_DIR`, etc. — near the top of
`main.py`; nothing else in the script needs to change.)*

---

## Results Summary

- **Dataset:** 10 people, 10 images each (100 total), varying expressions
- **Feature reduction:** 4,096 pixels → 42 PCA components (97.5× reduction, 95% variance retained)
- **Best model:** SVM (RBF kernel), C=10, gamma=0.01
- **Test accuracy:** 93.3% (28/30 correct on held-out test images)
- **Per-class performance:** perfect precision & recall on 8/10 people; minor confusion between 2 people (visible in confusion matrix)

---

## Requirements

```
numpy
pandas
scikit-learn
opencv-python-headless
matplotlib
seaborn
joblib
certifi
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Credits & References

- Dataset: AT&T Laboratories Cambridge — Olivetti Faces / ORL Database of Faces
- Method: Turk, M., & Pentland, A. (1991). *Eigenfaces for Recognition.* Journal of Cognitive Neuroscience.
- scikit-learn documentation: https://scikit-learn.org/stable/modules/decomposition.html#pca
