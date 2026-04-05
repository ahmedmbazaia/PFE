"""
Image classifier module.
Trains a Random Forest on 4 astrophysical source classes using synthetic
feature vectors. Features match those extracted by detect.py so the model
can be applied directly to detection results.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_PATH = os.path.join(DATA_DIR, "model.pkl")
OUTPUT_PLOT_DIR = os.path.join(DATA_DIR, "plots")
REPORT_PATH = os.path.join(OUTPUT_PLOT_DIR, "classification_report.txt")

FEATURE_COLS = ["area", "peak_brightness", "circularity", "compactness", "neighbor_distance"]

CLASS_NAMES = [
    "Étoile ponctuelle",
    "Système binaire",
    "Nébuleuse étendue",
    "Objet compact",
]


def generate_synthetic_features(n_per_class=300, seed=42):
    """
    Generate synthetic feature vectors for each of the 4 source classes.
    Distributions are calibrated to match real detection feature profiles.

    Features:
      area             — contour area in px²
      peak_brightness  — max pixel intensity in source ROI (0–255)
      circularity      — 4π·area/perimeter² (1.0 = perfect circle)
      compactness      — peak_brightness / area
      neighbor_distance — distance to nearest other source (px)
    """
    rng = np.random.RandomState(seed)
    records = []

    # ── Étoile ponctuelle: small, sharp, bright, high circularity ────────────
    for _ in range(n_per_class):
        area        = rng.uniform(10, 55)
        peak        = rng.uniform(150, 255)
        circ        = rng.uniform(0.65, 0.99)
        compactness = peak / (area + 1e-6)
        nd          = rng.uniform(30, 250)
        records.append([area, peak, circ, compactness, nd, "Étoile ponctuelle"])

    # ── Système binaire: close neighbour within 20 px ────────────────────────
    for _ in range(n_per_class):
        area        = rng.uniform(12, 60)
        peak        = rng.uniform(120, 220)
        circ        = rng.uniform(0.40, 0.82)
        compactness = peak / (area + 1e-6)
        nd          = rng.uniform(3, 19)          # defining feature: close pair
        records.append([area, peak, circ, compactness, nd, "Système binaire"])

    # ── Nébuleuse étendue: large, diffuse, low circularity ───────────────────
    for _ in range(n_per_class):
        area        = rng.uniform(100, 500)
        peak        = rng.uniform(40, 130)
        circ        = rng.uniform(0.05, 0.38)
        compactness = peak / (area + 1e-6)
        nd          = rng.uniform(50, 250)
        records.append([area, peak, circ, compactness, nd, "Nébuleuse étendue"])

    # ── Objet compact: tiny, extremely bright, highest compactness ───────────
    for _ in range(n_per_class):
        area        = rng.uniform(4, 22)
        peak        = rng.uniform(200, 255)
        circ        = rng.uniform(0.70, 1.00)
        compactness = peak / (area + 1e-6)
        nd          = rng.uniform(30, 200)
        records.append([area, peak, circ, compactness, nd, "Objet compact"])

    return pd.DataFrame(records, columns=FEATURE_COLS + ["label"])


def classify_detections(detections, model_data):
    """
    Re-classify a list of detection dicts using the trained model.
    Updates 'classification' and 'confidence' in-place.
    Returns the mutated list.
    """
    if not detections:
        return detections

    clf          = model_data["model"]
    feature_cols = model_data["features"]

    X = np.array([[d.get(f, 0.0) for f in feature_cols] for d in detections],
                 dtype=np.float64)
    preds = clf.predict(X)
    probs = clf.predict_proba(X)

    for det, cls, prob in zip(detections, preds, probs):
        prev_cls = det.get("classification", "")
        det["classification"] = cls
        det["confidence"] = round(float(prob.max()), 2)
        # Clear arabic_name if source is no longer a point star
        if cls != "Étoile ponctuelle" and prev_cls == "Étoile ponctuelle":
            det["arabic_name"] = ""

    return detections


def train_and_save():
    """Train Random Forest on synthetic features, save model.pkl and report."""
    os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    df = generate_synthetic_features(n_per_class=300)
    counts = df["label"].value_counts().to_dict()
    print(f"Synthetic dataset: {len(df)} samples — {counts}")

    X = df[FEATURE_COLS].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES,
                                   zero_division=0)
    print("\n--- Classification Report ---")
    print(report)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("4-Class Astrophysical Source Classifier — Random Forest\n")
        f.write("=" * 56 + "\n\n")
        f.write(f"Features: {', '.join(FEATURE_COLS)}\n")
        f.write(f"Training samples: {len(X_train)}  |  Test samples: {len(X_test)}\n\n")
        f.write(report)
    print(f"Classification report saved to {os.path.abspath(REPORT_PATH)}")

    # ── Feature importance plot ───────────────────────────────────────────────
    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1]
    palette = ["#3b82f6", "#22d3a0", "#f59e0b", "#ef4444", "#a78bfa"]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(range(len(FEATURE_COLS)),
           importances[idx],
           color=[palette[i % len(palette)] for i in range(len(FEATURE_COLS))])
    ax.set_xticks(range(len(FEATURE_COLS)))
    ax.set_xticklabels([FEATURE_COLS[i] for i in idx], rotation=35, ha="right", fontsize=9)
    ax.set_title("Feature Importances — 4-Class Source Classifier", fontsize=10, color="white")
    ax.set_ylabel("Importance", color="white")
    ax.set_facecolor("#0c1428")
    fig.patch.set_facecolor("#080e1c")
    ax.tick_params(colors="white")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#3d5068")
    fig.tight_layout()
    plot_path = os.path.join(OUTPUT_PLOT_DIR, "feature_importances.png")
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Feature importance plot saved to {os.path.abspath(plot_path)}")

    # ── Save model ────────────────────────────────────────────────────────────
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": clf, "features": FEATURE_COLS}, f)
    print(f"Model saved to {os.path.abspath(MODEL_PATH)}")

    return clf


def main():
    print("=== 4-Class Astrophysical Source Classifier ===")
    train_and_save()


if __name__ == "__main__":
    main()
