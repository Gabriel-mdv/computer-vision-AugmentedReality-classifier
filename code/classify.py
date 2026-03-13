"""
STARTER CODE for Binary Image Classification using FAST + BRIEF

Run this file after:
  1. Setting TARGET_SIZE and augmentation factors
  2. Verifying the dataset exists in data/classify/face/ and data/classify/no_face/
  3. Ensuring helper.py is in the SAME directory as this file
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from helper import corner_detection, computeBrief
from sklearn.ensemble import RandomForestClassifier


# ═══════════════════════════════════════════════════════════════════
#  WHAT THE HELPER FUNCTIONS GIVE YOU
# ═══════════════════════════════════════════════════════════════════
#
#  corner_detection(img, sigma=0.15)
#  ─────────────────────────────────
#    img    : 2D numpy array, uint8 grayscale (NOT float, NOT 3-channel)
#    sigma  : FAST corner threshold — higher means fewer corners detected
#    returns: locs  shape (K, 2)  dtype int64
#             Each row is [ROW, COL]  ← skimage convention, NOT OpenCV [x, y]
#             Returns shape (0, 2) when no corners are found — never None
#
#  computeBrief(img, locs)
#  ───────────────────────
#    img    : 2D uint8 grayscale (same image you passed to corner_detection)
#    locs   : (K, 2) int64 [row, col]  — pass directly from corner_detection
#             DO NOT swap to [col, row] or [x, y]
#    Filters keypoints within 4 pixels of any image border.
#    returns: desc  shape (K', 256)  dtype float64  values {0.0, 1.0}
#             locs  shape (K', 2)    border-filtered subset of input locs
#             K' may be 0 even when K > 0  (all points were near border)
#             D is ALWAYS 256 — hardcoded inside helper, never changes
#
#  CRITICAL: np.mean on a (0, 256) array returns a NaN vector.
#            Always check  if len(desc) == 0  BEFORE calling np.mean.
#
# ═══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════
#  DATASET STRUCTURE
# ═══════════════════════════════════════════════════════════════════
#
#  The Face vs No-Face dataset is provided in the assignment assets.
#  All images have been preprocessed to the same dimensions.
#
#  data/
#  └── classify/
#      ├── face/          ← images containing faces
#      │   ├── img_01.jpg
#      │   └── ...
#      └── no_face/       ← images without faces
#          ├── img_01.jpg
#          └── ...
#
# ═══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION  ← fill in BEFORE running anything
# ═══════════════════════════════════════════════════════════════════

# Every image is resized to this (height, width) before any processing.
# All images must share the same size — required by corner_detection.
#
# Q: Why must all images be resized to the same TARGET_SIZE before
#    extracting FAST keypoints?  (answer in extract_features below)
#
TARGET_SIZE = (256, 256)# TODO: replace with e.g. (128, 128)

# ── Augmentation variation factors ──────────────────────────────────
# Choose exactly 2 factors of variation to simulate through augmentation.
# These should be relevant to real-world variation in the Face vs No-Face
# classification task (e.g., 'Brightness', 'Rotation').
#
# See the factors table in the handout for options.
#
AUGMENTATION_FACTOR_1 = '???'  # TODO: e.g. 'Brightness'
AUGMENTATION_FACTOR_2 = '???'  # TODO: e.g. 'Rotation'

DATA_DIR    = '../data/classify'
RESULTS_DIR = '../results'

# D is hardcoded inside computeBrief as nbits=256 — do not change
BRIEF_D = 256


# ═══════════════════════════════════════════════════════════════════
#  1. LOAD DATASET  (provided — do not modify)
# ═══════════════════════════════════════════════════════════════════

def load_dataset(data_dir, target_size=TARGET_SIZE):
    """
    Loads, resizes, and converts every image to uint8 grayscale.

    Why uint8?
        corner_detection and computeBrief both call skimage functions that
        require a 2D uint8 array. Passing float images causes incorrect
        descriptor values. cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) after
        cv2.imread guarantees uint8.

    Returns
    -------
    images     : list of (H, W) ndarray, dtype uint8
    labels     : list of int
    class_names: list of str
    file_paths : list of str
    """
    images, labels, file_paths = [], [], []
    class_names = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])
    for idx, cls in enumerate(class_names):
        folder = os.path.join(data_dir, cls)
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            full_path = os.path.join(folder, fname)
            img = cv2.imread(full_path)
            if img is None:
                print(f"  [WARN] unreadable: {fname}")
                continue
            # resize → uint8 BGR, then convert to 2D uint8 grayscale
            resized = cv2.resize(img, (target_size[1], target_size[0]))
            gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            # gray.dtype == uint8, gray.ndim == 2  ✓
            images.append(gray)
            labels.append(idx)
            file_paths.append(os.path.abspath(full_path))

    print(f"Loaded {len(images)} images  size={target_size}  "
          f"dtype={images[0].dtype if images else 'N/A'}  classes={class_names}")
    return images, labels, class_names, file_paths


# ═══════════════════════════════════════════════════════════════════
#  2. DATASET GRID  (provided — do not modify)
# ═══════════════════════════════════════════════════════════════════

def show_dataset_grid(images, labels, class_names, file_paths,
                      n_per_class=4, save_dir=RESULTS_DIR):
    """
    Saves results/dataset_grid.png and prints each image's file path.
    Include both the grid image and the printed paths in your write-up.
    """
    os.makedirs(save_dir, exist_ok=True)
    n_cls = len(class_names)
    fig, axes = plt.subplots(n_cls, n_per_class,
                             figsize=(n_per_class * 2.8, n_cls * 3.0))
    print("\n========== DATASET IMAGE LINKS ==========")
    for ci, cls in enumerate(class_names):
        samples = [(im, fp) for im, lbl, fp
                   in zip(images, labels, file_paths) if lbl == ci][:n_per_class]
        print(f"\n  Class {ci} — '{cls}':")
        for j, (im, fp) in enumerate(samples):
            ax = axes[ci][j] if n_cls > 1 else axes[j]
            ax.imshow(im, cmap='gray')
            ax.axis('off')
            ax.set_title(os.path.basename(fp), fontsize=7, pad=3)
            if j == 0:
                ax.set_ylabel(cls, fontsize=11, fontweight='bold',
                              rotation=0, labelpad=55, va='center')
            print(f"    [{j+1}] {fp}")
    print("=========================================\n")
    plt.suptitle(
        f'Dataset Samples  {TARGET_SIZE[0]}×{TARGET_SIZE[1]} px\n'
        f'Dataset: Face vs No-Face',
        fontsize=11, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, 'dataset_grid.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.show()
    print(f"Dataset grid saved → {path}")


# ═══════════════════════════════════════════════════════════════════
#  3. DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────
# FACTOR SELECTION  ← complete before writing any code
#
# Factor 1 chosen: Brightness
#   Why relevant to Face vs No-Face classification (2–3 sentences):
#   Real-world face images are captured under varying lighting conditions,
#   from bright daylight to dim indoor lighting. A classifier trained only
#   on well-lit images may fail on darker or overexposed images, so
#   simulating brightness variation helps it generalise across conditions.
#   OpenCV function you will use:
#   np.clip(img.astype(np.int16) + shift, 0, 255).astype(np.uint8)
#   Parameter range (e.g., ±30 pixels, ±15°):
#   Random brightness shift in range [-40, +40] pixel intensity units.
#
# Factor 2 chosen: Flip
#   Why relevant to Face vs No-Face classification (2–3 sentences):
#   Faces can appear mirrored in images depending on camera orientation
#   or subject pose, and a classifier should treat left-facing and
#   right-facing faces equally. Horizontal flipping doubles the dataset
#   while fully preserving all facial structure and features.
#   OpenCV function you will use:
#   cv2.flip(img, 1)
#   Parameter range (e.g., ±30 pixels, ±15°):
#   Horizontal flip only (flip code = 1), applied to every image.
#
# Q: Name one factor that would HURT BRIEF descriptor matching
#    if used as augmentation and explain why.  (2–3 sentences)
#   Rotation would hurt BRIEF descriptor matching because BRIEF samples
#   pixel pairs at fixed pre-defined locations relative to each keypoint
#   without accounting for orientation. When an image is rotated, the
#   same keypoint produces a completely different binary descriptor string,
#   causing valid matches to be missed and false matches to increase.
# ───────────────────────────────────────────────────────────────────

AUGMENTATION_FACTOR_1 = 'Brightness'
AUGMENTATION_FACTOR_2 = 'Flip'

def augment_images(images, labels, seed=42):
    """
    Produces 2 augmented copies of every image — one per factor.
    Dataset grows from N to 3N.

    dtype contract
    ──────────────
    All augmented images MUST remain 2D uint8 arrays so that
    corner_detection and computeBrief accept them without modification.
    Verified safe operations:
        cv2.flip(img, 1)                         → uint8 ✓
        cv2.warpAffine(img, M, ...)              → uint8 ✓
        np.clip(..., 0, 255).astype(np.uint8)    → uint8 ✓

    Parameters
    ----------
    images : list of (H, W) uint8 ndarray
    labels : list of int
    seed   : int  (for reproducibility)

    Returns
    -------
    aug_images : list of (H, W) uint8 ndarray — originals + augmented
    aug_labels : list of int
    """
    np.random.seed(seed)
    aug_images = list(images)
    aug_labels = list(labels)
    h, w   = images[0].shape
    centre = (w // 2, h // 2)   # cv2 expects (col, row) for centre point

    for img, lbl in zip(images, labels):

        # ─────────────────────────────────────────────────────────
        # TODO — Augmentation for AUGMENTATION_FACTOR_1
        #
        # Simulate the variation in Factor 1.
        # The result must be a 2D uint8 array.
        #
        # Useful functions:
        #   Brightness : np.clip(img.astype(np.int16) + shift, 0, 255).astype(np.uint8)
        #   H-Flip     : cv2.flip(img, 1)
        #   Blur       : cv2.GaussianBlur(img, (k, k), sigma)
        #   Noise      : np.clip(img.astype(np.int16) + noise_array, 0, 255).astype(np.uint8)
        # ─────────────────────────────────────────────────────────
        shift = np.random.randint(-40, 40)
        aug1 = np.clip(img.astype(np.int16) + shift, 0, 255).astype(np.uint8)
        aug_images.append(aug1)
        aug_labels.append(lbl)

        # ─────────────────────────────────────────────────────────
        # TODO — Augmentation for AUGMENTATION_FACTOR_2
        #
        # Simulate the variation in Factor 2.
        # The result must be a 2D uint8 array.
        #
        # Useful functions:
        #   Rotation   : cv2.warpAffine(img, cv2.getRotationMatrix2D(centre, angle, 1.0),
        #                               (w, h), borderMode=cv2.BORDER_REFLECT)
        #   Scale/zoom : cv2.resize then crop/pad back to TARGET_SIZE
        #   Perspective: cv2.warpPerspective(img, M, (w, h))
        # ─────────────────────────────────────────────────────────
        aug2 = cv2.flip(img, 1)
        aug_images.append(aug2)
        aug_labels.append(lbl)

        pass  # ← remove once implemented

    n_orig = len(images)
    n_aug  = len(aug_images)
    print(f"Augmentation: {n_orig} → {n_aug}  (×{n_aug // n_orig})  "
          f"factors: {AUGMENTATION_FACTOR_1} + {AUGMENTATION_FACTOR_2}")
    return aug_images, aug_labels


def show_augmentation_examples(images, labels, class_names,
                                aug_images, aug_labels, save_dir=RESULTS_DIR):
    """Saves results/augmentation_examples.png. Include in your write-up."""
    os.makedirs(save_dir, exist_ok=True)
    n_cls  = len(class_names)
    n_orig = len(images)
    titles = ['Original', AUGMENTATION_FACTOR_1, AUGMENTATION_FACTOR_2]
    fig, axes = plt.subplots(n_cls, 3, figsize=(9, n_cls * 3.2))
    for ci, cls in enumerate(class_names):
        orig     = next(im for im, lbl in zip(images, labels) if lbl == ci)
        aug_imgs = [im for im, lbl
                    in zip(aug_images[n_orig:], aug_labels[n_orig:])
                    if lbl == ci][:2]
        for j, (var, title) in enumerate(zip([orig] + aug_imgs, titles)):
            ax = axes[ci][j] if n_cls > 1 else axes[j]
            ax.imshow(var, cmap='gray')
            ax.axis('off')
            ax.set_title(title, fontsize=10,
                         fontweight='bold' if j == 0 else 'normal')
            if j == 0:
                ax.set_ylabel(cls, fontsize=11, fontweight='bold',
                              rotation=0, labelpad=55, va='center')
    plt.suptitle(
        f'Augmentation  Factor 1: {AUGMENTATION_FACTOR_1}  '
        f'Factor 2: {AUGMENTATION_FACTOR_2}',
        fontsize=11, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, 'augmentation_examples.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.show()
    print(f"Augmentation grid saved → {path}")


# ═══════════════════════════════════════════════════════════════════
#  4. EXTRACT FEATURES
# ═══════════════════════════════════════════════════════════════════

def extract_features(images, sigma=0.15):
    """
    FAST keypoints  →  BRIEF descriptors  →  mean pooling  →  (N, 256) matrix

    Parameters
    ----------
    images : list of (H, W) uint8 ndarray — must be 2D uint8 grayscale
    sigma  : FAST threshold passed to corner_detection (default 0.15)

    Returns
    -------
    feature_matrix : (N, 256) float64 ndarray — one row per image, no NaN
    kp_counts      : list of int — raw keypoint count per image
    """

  # ─────────────────────────────────────────────────────────────
    # CONCEPTUAL QUESTION 1 — answer before coding
    # Q: Why must all images be resized to TARGET_SIZE before calling
    #    corner_detection?  What would go wrong with different sizes?
    #    (2–3 sentences)
    #    All images must be the same size so that the feature matrix
    #    has a consistent shape (N, 256) — one fixed-length vector per
    #    image. If images had different sizes, corner_detection would
    #    detect different numbers of keypoints and at different scales,
    #    making the extracted features incomparable across images and
    #    causing the classifier to receive inconsistent input dimensions.
    # ─────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────
    # CONCEPTUAL QUESTION 2 — answer before coding
    # Q: computeBrief returns a (K', 256) matrix — one row per keypoint.
    #    Why can't you pass this directly to a classifier?
    #    How does mean pooling fix this?  (2–3 sentences)
    #    A classifier requires exactly one fixed-length feature vector
    #    per image, but different images produce different numbers of
    #    keypoints K', so the descriptor matrix has a variable number
    #    of rows. Mean pooling collapses the (K', 256) matrix into a
    #    single (256,) vector by averaging across all keypoints, giving
    #    every image the same fixed-length representation regardless of
    #    how many keypoints were detected.
    # ─────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────
    # CONCEPTUAL QUESTION 3 — answer before coding
    # Q: What information is lost when you mean-pool K' descriptors
    #    into one vector?  How might this hurt accuracy?  (2–3 sentences)
    #    Mean pooling discards spatial information — the locations of
    #    individual keypoints and the relationships between them are
    #    lost. Two images with very different spatial arrangements of
    #    features could produce similar mean-pooled vectors, making
    #    them appear identical to the classifier. This loss of spatial
    #    structure may hurt accuracy on images where the position of
    #    facial features is the key discriminating factor.
    # ─────────────────────────────────────────────────────────────

    feature_matrix = []
    kp_counts      = []

    for img in images:

        # ─────────────────────────────────────────────────────────
        # TODO Step 1 — detect keypoints
        #
        # Call corner_detection(img, sigma).
        # It returns locs: (K, 2) int64 [row, col].
        # If K == 0, append np.zeros(BRIEF_D) to feature_matrix,
        # append 0 to kp_counts, and continue to the next image.
        # ─────────────────────────────────────────────────────────
        locs = corner_detection(img, sigma)
        if len(locs) == 0:
            feature_matrix.append(np.zeros(BRIEF_D))
            kp_counts.append(0)
            continue

        # ─────────────────────────────────────────────────────────
        # TODO Step 2 — compute BRIEF descriptors
        #
        # Call computeBrief(img, locs).
        # Pass locs EXACTLY as returned by corner_detection — do NOT
        # swap rows/columns or convert to OpenCV [x, y] format.
        # Returns desc: (K', 256) float64  and  locs: (K', 2).
        #
        # CRITICAL: if len(desc) == 0, append np.zeros(BRIEF_D) and
        # continue.  Do NOT call np.mean on an empty array — it returns
        # NaN which will silently corrupt your classifier.
        # ─────────────────────────────────────────────────────────
        desc, locs = computeBrief(img, locs)
        if len(desc) == 0:
            feature_matrix.append(np.zeros(BRIEF_D))
            kp_counts.append(0)
            continue

        # ─────────────────────────────────────────────────────────
        # TODO Step 3 — mean pool (K', 256) → (256,)
        #
        # feature_vector = np.mean(desc, axis=0)
        # Append to feature_matrix and record len(locs) in kp_counts.
        # ─────────────────────────────────────────────────────────
        feature_vector = np.mean(desc, axis=0)
        feature_matrix.append(feature_vector)
        kp_counts.append(len(locs))

    feature_matrix = np.vstack(feature_matrix)  # (N, 256)
    print(f"Feature matrix: {feature_matrix.shape}  "
          f"NaN present: {np.isnan(feature_matrix).any()}  "
          f"avg kp/img: {np.mean(kp_counts):.1f}")
    return feature_matrix, kp_counts

# ═══════════════════════════════════════════════════════════════════
#  5. TRAIN / TEST SPLIT  (provided — do not modify)
# ═══════════════════════════════════════════════════════════════════

def train_test_split(feature_matrix, labels, train_ratio=0.8, seed=42):
    np.random.seed(seed)
    labels         = np.array(labels)
    feature_matrix = np.array(feature_matrix)
    X_train, X_test, y_train, y_test = [], [], [], []
    for cls in np.unique(labels):
        idx     = np.where(labels == cls)[0]
        np.random.shuffle(idx)
        n_train = max(1, int(len(idx) * train_ratio))
        X_train.append(feature_matrix[idx[:n_train]])
        X_test.append(feature_matrix[idx[n_train:]])
        y_train.extend(labels[idx[:n_train]].tolist())
        y_test.extend(labels[idx[n_train:]].tolist())
    X_train, X_test = np.vstack(X_train), np.vstack(X_test)
    y_train, y_test = np.array(y_train),  np.array(y_test)
    print(f"Split: train={len(y_train)}  test={len(y_test)}  ratio={train_ratio}")
    return X_train, X_test, y_train, y_test

# ═══════════════════════════════════════════════════════════════════
#  6. TRAIN CLASSIFIER
# ═══════════════════════════════════════════════════════════════════

def train_classifier(X_train, y_train):
    """
    Trains one classifier on the (N, 256) feature matrix.

    ┌────────────────────────────────────────────────────────────┐
    │  CONSTRAINT: SVM / SVC is NOT permitted.                   │
    │  Choose ONE of:                                            │
    │    (a) sklearn.neighbors.KNeighborsClassifier              │
    │    (b) sklearn.ensemble.RandomForestClassifier             │
    └────────────────────────────────────────────────────────────┘

    Parameters
    ----------
    X_train : (N_train, 256) float64 ndarray
    y_train : (N_train,)     int ndarray

    Returns
    -------
    classifier : trained sklearn classifier
    """

    # ─────────────────────────────────────────────────────────────
    # CONCEPTUAL QUESTION 4 — answer before coding
    # Q: Which classifier did you choose?
    #    How does it produce a prediction from a (256,) feature vector?
    #    Why is it appropriate for a small dataset?  (3–4 sentences)

    # We chose the Random Forest classifier. During training, it learns an
    # ensemble of decision trees, each built from a random subset of the
    # training samples and feature dimensions. Given a new (256,) feature
    # vector, the classifier passes it through all trees, where each tree
    # predicts a class label based on learned decision rules, and the final
    # prediction is obtained by majority voting across the trees. Random
    # Forest is appropriate for a small dataset because bootstrap sampling
    # and feature randomness help reduce overfitting, making the model more
    # robust when training data is limited.
    # ─────────────────────────────────────────────────────────────

    # TODO: import your chosen classifier
    # from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier

    # TODO: instantiate with your hyperparameters and justify each one
    # n_estimators=100: 100 trees gives stable predictions without
    # being computationally expensive on a small dataset.
    # random_state=42: ensures reproducibility across runs.
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # TODO: fit and return
    classifier.fit(X_train, y_train)
    return classifier

# ═══════════════════════════════════════════════════════════════════
#  7. EVALUATE  (provided — do not modify)
# ═══════════════════════════════════════════════════════════════════

def evaluate(classifier, X_test, y_test, class_names, save_dir=RESULTS_DIR):
    from sklearn.metrics import (accuracy_score, precision_score,
                                  recall_score, f1_score,
                                  confusion_matrix, ConfusionMatrixDisplay)
    os.makedirs(save_dir, exist_ok=True)
    y_pred    = classifier.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
    recall    = recall_score(y_test,    y_pred, average='binary', zero_division=0)
    f1        = f1_score(y_test,        y_pred, average='binary', zero_division=0)
    cm        = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  Image size  : {TARGET_SIZE[0]}×{TARGET_SIZE[1]}  D={BRIEF_D}")
    print(f"  Classes     : {class_names}")
    print(f"  Test samples: {len(y_test)}")
    print(f"  Accuracy    : {accuracy:.4f}")
    print(f"  Precision   : {precision:.4f}")
    print(f"  Recall      : {recall:.4f}")
    print(f"  F1 Score    : {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print(f"{'='*50}\n")

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=class_names).plot(
        cmap=plt.cm.Blues, ax=ax, colorbar=False)
    ax.set_title('Confusion Matrix', fontsize=11)
    plt.tight_layout()
    path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(path, dpi=120)
    plt.show()
    print(f"Confusion matrix saved → {path}")

    return dict(accuracy=accuracy, precision=precision,
                recall=recall, f1=f1, confusion_matrix=cm)


# ═══════════════════════════════════════════════════════════════════
#  8. TEST ON YOUR OWN IMAGE
# ═══════════════════════════════════════════════════════════════════
def predict_single_image(image_path, classifier):
    # ─────────────────────────────────────────────────────────────
    # TODO — implement inference pipeline
    #
    # 1. Load image with cv2.imread
    # 2. Resize to TARGET_SIZE
    # 3. Convert to grayscale (2D uint8)
    # 4. Extract features using extract_features([img])
    # 5. Predict using classifier.predict
    # 6. Get confidence using classifier.predict_proba if available
    # ─────────────────────────────────────────────────────────────
    # 1. Load image
    img = cv2.imread(image_path)

    # 2. Resize to TARGET_SIZE
    img = cv2.resize(img, (TARGET_SIZE[1], TARGET_SIZE[0]))

    # 3. Convert to grayscale (2D uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 4. Extract features
    feature_matrix, _ = extract_features([img])

    # 5. Predict
    prediction = classifier.predict(feature_matrix)[0]
    class_names = ['face', 'no_face']
    class_name = class_names[prediction]

    # 6. Get confidence using predict_proba if available
    if hasattr(classifier, 'predict_proba'):
        confidence = classifier.predict_proba(feature_matrix)[0][prediction]
    else:
        confidence = 1.0

    print(f"Predicted class: {class_name}  Confidence: {confidence:.2%}")
    return prediction, class_name, confidence


# ═══════════════════════════════════════════════════════════════════
#  MAIN  —  full pipeline
# ═══════════════════════════════════════════════════════════════════

def main():
    np.random.seed(42)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. load — all images are 2D uint8 grayscale after this call
    images, labels, class_names, file_paths = load_dataset(DATA_DIR, TARGET_SIZE)
    print(f"Image: shape={images[0].shape}  dtype={images[0].dtype}")

    # 2. dataset grid with file links
    show_dataset_grid(images, labels, class_names, file_paths,
                      n_per_class=4, save_dir=RESULTS_DIR)

    # 3. augment — images must remain 2D uint8 after augmentation
    aug_images, aug_labels = augment_images(images, labels)
    show_augmentation_examples(images, labels, class_names,
                                aug_images, aug_labels, RESULTS_DIR)
    
    # aug_images, aug_labels = images, labels

    # 4. extract — uses corner_detection + computeBrief from helper.py
    print("\nExtracting features ...")
    feature_matrix, kp_counts = extract_features(aug_images, sigma=0.15)
    print(f"Feature matrix shape: {feature_matrix.shape}  "
          f"(should be (N, {BRIEF_D})  NaN={np.isnan(feature_matrix).any()})")

    # 5. split
    # TODO: justify your train_ratio choice in your write-up
    train_ratio = 0.7
    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix, aug_labels, train_ratio=train_ratio)

    # 6. train
    print("\nTraining classifier ...")
    classifier = train_classifier(X_train, y_train)

    # 7. evaluate
    results = evaluate(classifier, X_test, y_test, class_names)
    
    # 8. Find and display misclassified images
    y_pred = classifier.predict(X_test)

    # Get test indices the same way train_test_split does
    np.random.seed(42)
    labels_array = np.array(aug_labels)
    test_indices = []
    for cls in np.unique(labels_array):
        idx = np.where(labels_array == cls)[0]
        np.random.shuffle(idx)
        n_train = max(1, int(len(idx) * train_ratio))
        test_indices.extend(idx[n_train:].tolist())

    # Display misclassified images
    for i, (true, pred) in enumerate(zip(y_test, y_pred)):
        if true != pred:
            actual_idx = test_indices[i]
            misclassified_img = aug_images[actual_idx]
            print(f"True label: {class_names[true]}, Predicted: {class_names[pred]}")
            plt.figure()
            plt.imshow(misclassified_img, cmap='gray')
            plt.title(f"True: {class_names[true]}  |  Predicted: {class_names[pred]}")
            plt.axis('off')
            plt.show()

   # ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
    # PIPELINE DOCUMENTATION TASK  ← fill in after running
    #
    # Report these values in your write-up under Part D:
    #
    # TARGET_SIZE (H × W)             : 256 × 256
    # Augmentation Factor 1           : Brightness
    # Augmentation Factor 2           : Flip
    # Total images (before aug)       : 30
    # Total images (after  aug)       : 90
    # Descriptor dim D                : 256  (always 256)
    # Feature matrix shape (N, D)     : (90, 256)
    # Train samples                   : 62
    # Test samples                    : 28
    # Classifier used                 : Random Forest
    # Key hyperparameter              : n_estimators = 100
    # ─────────────────────────────────────────────────────────────
    
    # ─────────────────────────────────────────────────────────────
    # TEST ON YOUR OWN IMAGE TASK  ← complete for Task D5
    #
    # Uncomment and fill in after implementing predict_single_image:
    #
    test_image_path = "../data/photo.png"
    prediction, class_name, confidence = predict_single_image(
        test_image_path, classifier)
    print(f"\nTest Image Prediction:")
    print(f"  Predicted class: {class_name}")
    print(f"  Confidence: {confidence:.2%}")
    # ─────────────────────────────────────────────────────────────

    return results


if __name__ == '__main__':
    main()
