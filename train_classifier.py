"""
train_classifier.py
--------------------
Builds a labelled shape dataset OURSELVES (no external / pre-trained
data or models), extracts geometric + Hu-moment features, and trains
a lightweight classifier (K-Nearest-Neighbours) completely from
scratch. This satisfies the "pattern recognition / classification"
requirement of the IPPR guideline without touching any pre-trained
ML/DL model.

Run:
    python3 train_classifier.py

Outputs (all needed for the report's "Testing" chapter):
    dataset/                     -> generated shape images (train+test)
    models/shape_classifier.pkl  -> trained KNN model
    models/scaler.pkl            -> feature scaler
    models/label_encoder.pkl     -> label encoder
    outputs/confusion_matrix.png
    outputs/accuracy_report.txt
"""

import os
import cv2
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shape_utils import extract_features, FEATURE_NAMES

SHAPES = ["Circle", "Square", "Rectangle", "Triangle", "Pentagon", "Hexagon", "Star"]
IMG_SIZE = 300
SAMPLES_PER_CLASS = 120  # -> 7 * 120 = 840 total images (well above the 100-min requirement)

DATASET_DIR = "dataset"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"
for d in (DATASET_DIR, MODEL_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)


def random_point(cx, cy, r, angle):
    return (int(cx + r * np.cos(angle)), int(cy + r * np.sin(angle)))


def draw_regular_polygon(img, cx, cy, r, n_sides, rotation, color, thickness):
    pts = []
    for i in range(n_sides):
        angle = rotation + i * (2 * np.pi / n_sides)
        pts.append(random_point(cx, cy, r, angle))
    pts = np.array(pts, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], color)
    return img


def draw_star(img, cx, cy, r_outer, r_inner, rotation, color):
    pts = []
    for i in range(10):
        r = r_outer if i % 2 == 0 else r_inner
        angle = rotation + i * (2 * np.pi / 10)
        pts.append(random_point(cx, cy, r, angle))
    pts = np.array(pts, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], color)
    return img


def generate_shape_image(shape_name, idx):
    """Create one synthetic training image with random size/rotation/
    position/noise so the classifier learns to be robust."""
    img = np.full((IMG_SIZE, IMG_SIZE, 3), 255, np.uint8)
    cx = IMG_SIZE // 2 + np.random.randint(-20, 20)
    cy = IMG_SIZE // 2 + np.random.randint(-20, 20)
    r = np.random.randint(70, 110)
    rotation = np.random.uniform(0, 2 * np.pi)
    color = (0, 0, 0)

    if shape_name == "Circle":
        cv2.circle(img, (cx, cy), r, color, -1)
    elif shape_name == "Square":
        s = r
        pts = cv2.boxPoints(((cx, cy), (s, s), np.degrees(rotation)))
        cv2.fillPoly(img, [np.int32(pts)], color)
    elif shape_name == "Rectangle":
        w, h = r, int(r * np.random.uniform(0.5, 0.75))
        pts = cv2.boxPoints(((cx, cy), (w, h), np.degrees(rotation)))
        cv2.fillPoly(img, [np.int32(pts)], color)
    elif shape_name == "Triangle":
        draw_regular_polygon(img, cx, cy, r, 3, rotation, color, -1)
    elif shape_name == "Pentagon":
        draw_regular_polygon(img, cx, cy, r, 5, rotation, color, -1)
    elif shape_name == "Hexagon":
        draw_regular_polygon(img, cx, cy, r, 6, rotation, color, -1)
    elif shape_name == "Star":
        draw_star(img, cx, cy, r, int(r * 0.45), rotation, color)

    # add a bit of Gaussian noise to simulate real camera conditions
    noise = np.random.normal(0, 6, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    fname = f"{DATASET_DIR}/{shape_name}_{idx}.png"
    cv2.imwrite(fname, img)
    return img


def build_dataset():
    X, y = [], []
    for shape in SHAPES:
        for i in range(SAMPLES_PER_CLASS):
            img = generate_shape_image(shape, i)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            c = max(contours, key=cv2.contourArea)
            result = extract_features(c)
            if result is None:
                continue
            features, _ = result
            X.append(features)
            y.append(shape)
    return np.array(X), np.array(y)


def main():
    print("Generating synthetic dataset...")
    X, y = build_dataset()
    print(f"Dataset built: {X.shape[0]} samples, {X.shape[1]} features each.")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    report_txt = classification_report(y_test, y_pred, target_names=le.classes_)
    print(report_txt)
    print(f"Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")

    with open(f"{OUTPUT_DIR}/accuracy_report.txt", "w") as f:
        f.write(f"Overall Accuracy : {acc:.4f}\n")
        f.write(f"Macro Precision  : {prec:.4f}\n")
        f.write(f"Macro Recall     : {rec:.4f}\n")
        f.write(f"Macro F1-score   : {f1:.4f}\n\n")
        f.write(report_txt)

    # Confusion matrix plot (goes straight into the report)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(le.classes_)))
    ax.set_yticks(range(len(le.classes_)))
    ax.set_xticklabels(le.classes_, rotation=45, ha="right")
    ax.set_yticklabels(le.classes_)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix - Shape Classifier")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150)
    print(f"Saved confusion matrix -> {OUTPUT_DIR}/confusion_matrix.png")

    joblib.dump(model, f"{MODEL_DIR}/shape_classifier.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(le, f"{MODEL_DIR}/label_encoder.pkl")
    print(f"Saved model -> {MODEL_DIR}/shape_classifier.pkl")


if __name__ == "__main__":
    main()
