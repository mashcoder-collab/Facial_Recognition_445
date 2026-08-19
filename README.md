# Face Recognition System with Unknown-Person Detection

This is our machine learning project where we built a face recognition system using classical ML instead of deep learning. We trained and compared three models (KNN, SVM, Random Forest) on the AT&T (ORL) face dataset, then built a live webcam version that can recognize the people it was trained on and say "Unknown" for anyone else.

## Contributors

| Student ID | Name | GitHub |
|---|---|---|
| 2232167042 | Mehedi Hasan Dip | @Dip223 |
| 2321942642 | Sayef Ali Khan   | @mashcoder-collab |

CSE445 - Machine Learning, Section 7, Group 5, North South University

## Project Overview

### Part 1: Comparing Models
We used the AT&T (ORL) Face Dataset - 40 people, 10 grayscale images each, 400 images total. We treated the first 10 people as our "enrolled" users, and kept the other 30 aside as people the system has never seen, so we could properly test if it rejects strangers instead of just guessing.

- Controlled lighting, mostly front-facing, 92x112 px, grayscale
- 80/20 train/test split, plus 5-fold cross-validation to double check
- Trained KNN, SVM (linear kernel), and Random Forest (100 trees)

### Part 2: Making It Work Live
After comparing the models, we built a webcam version of the system:
- Face detection using Haar Cascade
- KNN matching, but with a separate distance threshold per person so it can say "Unknown" instead of always picking the closest match
- A voting system over the last 8 frames so the name on screen doesn't keep flickering
- A small calibration tool for faces captured through a phone camera, since those looked noticeably different from the original dataset photos

## Results

### Model Comparison (80/20 split)

| Model | Accuracy | Precision | Recall | F1 Score | 5-Fold CV |
|---|---|---|---|---|---|
| KNN (K=3) | 95.00% | 96.67% | 95.00% | 94.67% | 96.00% ± 3.74% |
| SVM (linear) | 100.00% | 100.00% | 100.00% | 100.00% | 99.00% ± 2.00% |
| Random Forest | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% ± 0.00% |

SVM and Random Forest actually scored a bit higher than KNN here. We still picked KNN for the live system though, because it gives us an actual distance value we can use to reject unknown faces. SVM and Random Forest don't give us that in any simple way.

### Rejecting Unknown Faces

| Group | Total Images | Correct | Accuracy |
|---|---|---|---|
| Enrolled faces | 100 | 100 | 100% |
| Stranger faces (correctly marked Unknown) | 300 | 300 | 100% |

Tested offline using all 100 images of our 10 enrolled people and all 300 images from the 30 people never used in training.

### Speed (tested on CPU, no GPU)

| Step | Time |
|---|---|
| Face detection (Haar Cascade) | ~24.85 ms |
| Preprocessing + KNN matching | ~12.68 ms |
| Total per frame | ~37.52 ms (around 27 FPS) |

## Project Structure

```
Face-Recognition-System/
│
├── dataset/                        # AT&T (ORL) Face Dataset
│   └── s1/ ... s40/                 # 10 grayscale .pgm images per person
│
├── models/                         # Saved trained models
│   ├── face_recognizer.pkl          # The deployed KNN model + thresholds
│   └── random_forest_model.pkl
│
├── results/
│   └── model_comparison.png         # Accuracy comparison chart
│
├── src/
│   ├── load_dataset.py              # Loads and labels the images
│   ├── train_knn.py
│   ├── train_svm.py
│   ├── train_random_forest.py
│   ├── evaluate_models.py           # 80/20 comparison: accuracy, precision, recall, F1
│   ├── cross_validation.py          # 5-fold CV comparison
│   ├── plot_results.py              # Makes the comparison chart
│   ├── train_face_recognizer.py     # Builds the deployed KNN + calibrates thresholds
│   ├── calibrate_phone.py           # Collects phone-camera samples
│   └── face_recognition_webcam.py   # The live webcam recognizer
│
├── main.py
├── requirements.txt
└── README.md
```

## Setup

### What you need
- Python 3.9+
- A webcam, if you want to run the live part

### 1. Clone the repo
```bash
git clone <your-repo-url>
cd Face-Recognition-System
```

### 2. Set up a virtual environment
**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```
**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install the requirements
```bash
pip install -r requirements.txt
```

### 4. Get the AT&T dataset
Download the [AT&T (ORL) Database of Faces](https://cam-orl.co.uk/facedatabase.html) and put it here:
```
dataset/
    s1/  1.pgm  2.pgm  ...  10.pgm
    s2/  1.pgm  2.pgm  ...  10.pgm
    ...
    s40/ 1.pgm  2.pgm  ...  10.pgm
```

## Running It

### Train and compare the three models
```bash
python -m src.train_knn
python -m src.train_svm
python -m src.train_random_forest
python -m src.evaluate_models        # accuracy, precision, recall, F1 on the 80/20 split
python -m src.cross_validation       # 5-fold CV
python -m src.plot_results           # builds results/model_comparison.png
```

### Set up the live recognizer
```bash
python -m src.train_face_recognizer  # trains KNN and calibrates per-person thresholds
python -m src.calibrate_phone        # optional - adds phone-camera samples
```

### Run it live
```bash
python -m src.face_recognition_webcam
```
Green means confirmed match, yellow means it's still checking, red means Unknown.

## How It Actually Works

**Training/comparison side:** `load_dataset.py` loads and labels everything, then `evaluate_models.py` and `cross_validation.py` handle training and testing the three models. `plot_results.py` turns the results into a chart.

**Webcam side:** for every frame, `face_recognition_webcam.py` does this:
1. Detects the face with Haar Cascade
2. Resizes it to 50x50, equalizes the histogram, normalizes it
3. Finds the closest match using KNN
4. Checks if that distance is within the threshold for that person - if it's too far, marks it Unknown
5. Waits until 5 out of the last 8 frames agree before actually showing a name on screen

## Notes on a Few Design Choices

We didn't use PCA, just raw pixel values after resizing and normalizing. Wanted to keep things simple and stick to the classical techniques covered in the course.

Instead of one global threshold for accepting or rejecting a face, we ended up calibrating a separate threshold for each person, since some people's photos naturally vary more than others (our thresholds ranged from about 0.47 to 0.70 across the 10 people). We actually tried a single global threshold first and it didn't work well - it was either too strict on real users or too loose on strangers, there wasn't a good middle ground.

KNN wasn't technically the best-performing model, but it was the only one that gave us a distance value we could actually use to detect unknown faces, so that's what we deployed.

We used the AT&T dataset instead of real student photos mainly because we didn't have the time or an approved way to collect real face data in the one week we had.

We also built a small tool to capture training images through a phone camera, since the original dataset photos look pretty different from what a webcam actually picks up in real conditions.

## Dependencies

```
opencv-python
scikit-learn
numpy
joblib
matplotlib
```
```bash
pip install -r requirements.txt
```

## About the Dataset

**AT&T (ORL) Database of Faces**
- 400 images, 40 people, 10 images each
- 92x112 px, grayscale
- Controlled lighting, mostly frontal, some pose variation
- We used the first 10 people as our enrolled set, and the rest to test unknown-face rejection
- Reference: F. S. Samaria and A. C. Harter, "Parameterisation of a stochastic model for human face identification," Proc. 2nd IEEE Workshop on Applications of Computer Vision, 1994.

## Acknowledgments

Made for CSE445 (Machine Learning) at North South University. We used Claude (Anthropic) for coding help and to draft parts of this documentation - the code, training, and results are all ours.
