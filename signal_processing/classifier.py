"""
Image classifier module.
Trains a Random Forest classifier to categorize images as 'star' or 'fringe'
based on extracted features. Saves the trained model as model.pkl.
"""

import os
import glob
import pickle
import numpy as np
import cv2
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
MODEL_PATH = os.path.join(DATA_DIR, "model.pkl")
OUTPUT_PLOT_DIR = os.path.join(DATA_DIR, "plots")


def extract_features(image):
    """Extract numerical features from a grayscale image for classification."""
    img = image.astype(np.float64)

    # Intensity statistics
    mean_val = np.mean(img)
    std_val = np.std(img)
    skew = float(pd.Series(img.ravel()).skew())
    kurtosis = float(pd.Series(img.ravel()).kurtosis())

    # FFT features - dominant frequency and power
    fft2 = np.fft.fft2(img)
    fft_mag = np.abs(np.fft.fftshift(fft2))
    cy, cx = fft_mag.shape[0] // 2, fft_mag.shape[1] // 2
    fft_mag[cy-2:cy+3, cx-2:cx+3] = 0  # remove DC
    total_power = np.sum(fft_mag)
    peak_power = np.max(fft_mag)
    fft_ratio = peak_power / (total_power + 1e-10)

    # Texture: Laplacian variance (sharpness)
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()

    # Number of bright spots (proxy for star count)
    _, binary = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n_blobs = len(contours)

    # Horizontal vs vertical gradient energy
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    grad_ratio = np.sum(np.abs(grad_x)) / (np.sum(np.abs(grad_y)) + 1e-10)

    return {
        "mean": mean_val,
        "std": std_val,
        "skew": skew,
        "kurtosis": kurtosis,
        "fft_peak_ratio": fft_ratio,
        "laplacian_var": laplacian_var,
        "n_blobs": n_blobs,
        "grad_ratio": grad_ratio,
    }


def load_dataset():
    """Load images and extract features with labels."""
    records = []
    for pattern, label in [("star_*.png", "star"), ("fringe_*.png", "fringe")]:
        files = sorted(glob.glob(os.path.join(IMAGES_DIR, pattern)))
        for fpath in files:
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            feats = extract_features(img)
            feats["label"] = label
            feats["file"] = os.path.basename(fpath)
            records.append(feats)
    return pd.DataFrame(records)


def train_and_save():
    """Train Random Forest classifier and save model."""
    os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

    df = load_dataset()
    if df.empty:
        print("No images found. Run synthetic_data.py first.")
        return

    print(f"Dataset: {len(df)} images ({df['label'].value_counts().to_dict()})")

    feature_cols = [c for c in df.columns if c not in ("label", "file")]
    X = df[feature_cols].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=["star", "fringe"])
    print("Confusion Matrix:")
    print(cm)

    # Feature importance plot
    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(feature_cols)), importances[idx], color="teal")
    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels([feature_cols[i] for i in idx], rotation=45, ha="right")
    ax.set_title("Random Forest Feature Importances")
    ax.set_ylabel("Importance")
    fig.tight_layout()
    plot_path = os.path.join(OUTPUT_PLOT_DIR, "feature_importances.png")
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Feature importance plot: {plot_path}")

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": clf, "features": feature_cols}, f)
    print(f"Model saved to {os.path.abspath(MODEL_PATH)}")

    return clf


def main():
    print("=== Image Classifier Training ===")
    train_and_save()


if __name__ == "__main__":
    main()
