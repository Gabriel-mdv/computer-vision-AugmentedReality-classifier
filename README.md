# AR with Planar Homographies & Image Classification

A computer vision pipeline that builds an Augmented Reality application using planar homographies, and a binary image classifier using handcrafted features  all from scratch using Python, OpenCV, and scikit-learn.

---

## What this does

### Augmented Reality
Tracks a book cover across video frames and overlays a video source onto it in real time. The book is detected automatically in each frame using feature matching, a homography is estimated to find the perspective transform, and the overlay video is warped and composited onto the book.

### Image Classification
A traditional machine learning pipeline that classifies images as **face** or **no face** without any deep learning. Uses FAST corner detection and BRIEF binary descriptors to extract features, mean pooling to produce fixed-length vectors, and a Random Forest classifier to make predictions.

---

## Tech Stack

- **Python**
- **OpenCV** - image processing, warping, feature detection
- **scikit-image** - FAST detector, BRIEF descriptor
- **scikit-learn** - Random Forest classifier, evaluation metrics
- **NumPy** - matrix operations, SVD
- **SciPy** - image rotation
- **Matplotlib** - visualizations

---

## Project Structure

`
code/
 matchPics.py           (FAST + BRIEF feature matching)
 planarH.py             (Homography: DLT, normalization, RANSAC)
 HarryPotterize.py      (Image overlay using homography)
 ar.py                  (Full AR video pipeline)
 briefRotTest.py        (BRIEF rotation sensitivity analysis)
 classify.py            (Full classification pipeline)
 loadVid.py             (Video loader)
 helper.py              (FAST + BRIEF helper functions)

results/
 ar.avi                 (AR output video)
 dataset_grid.png       (Dataset samples)
 augmentation_examples.png
 confusion_matrix.png
`

---

## Getting Started

Install dependencies:
`
pip install numpy opencv-python scikit-image scikit-learn scipy matplotlib
`

### Run the AR application
`
python ar.py
`

### Run the image classifier
`
python classify.py
`

### Test feature matching
`
python HarryPotterize.py
`

### Analyse BRIEF under rotation
`
python briefRotTest.py
`

---

## How it works

### Homography Pipeline
1. Detect FAST keypoints in both images
2. Compute BRIEF descriptors and match using Hamming distance
3. Estimate homography using Direct Linear Transform (DLT) + SVD
4. Refine with RANSAC to reject outlier matches
5. Warp and composite the overlay image

### Classification Pipeline
1. Load images and resize to 256x256
2. Apply data augmentation (brightness + horizontal flip)
3. Extract FAST keypoints and BRIEF descriptors per image
4. Mean-pool descriptors into a fixed 256-dimensional vector
5. Train a Random Forest classifier
6. Evaluate on held-out test set

---

## Results

| Metric | Value |
|--------|-------|
| AR tracking | Stable across translation in video frames |
| Classification accuracy | 96.43% (with augmentation) |
| Without augmentation | 80.00% |
| Precision | 1.0000 |
| Recall | 0.9286 |

---

## Interesting findings

- **BRIEF is not rotation invariant** - match count drops sharply beyond 20 degrees of rotation and only recovers near 360 degrees
- **Background clutter hurts classification** - a full-body photo was misclassified as no-face because background corners dominated the mean-pooled descriptor; cropping to the face fixed it
- **Augmentation matters** - tripling the dataset with brightness and flip augmentation improved accuracy from 80% to 96.43%

---

## Author
**jmpuhwez** - CMU-Africa, 2026
