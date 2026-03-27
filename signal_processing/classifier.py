#!/usr/bin/env python3
"""
Signal Quality Classifier
==========================
Random Forest classifier that categorizes interferometric signal
quality as GOOD, DEGRADED, or LOST based on visibility, phase,
baseline distance, and intensity features.

Can train on synthetic data and save/load a model to model.pkl.
"""

import os
import random

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ─── Paths ───────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# ─── Class labels ────────────────────────────────────────────
CLASSES = ["GOOD", "DEGRADED", "LOST"]

# ─── Model ───────────────────────────────────────────────────
_model = None


def generate_training_data(n_samples=1000):
    """
    Generate synthetic training data for the classifier.

    Features: [visibility, phase_rad, baseline_mm, t1_intensity, t2_intensity]
    Labels:   GOOD (V > 0.6), DEGRADED (0.2 < V ≤ 0.6), LOST (V ≤ 0.2)

    Adds realistic noise to make the boundary non-trivial.
    """
    X = []
    y = []

    for _ in range(n_samples):
        # Random ground truth class
        cls = random.choice(CLASSES)

        if cls == "GOOD":
            vis = random.uniform(0.55, 1.0)
            t1 = random.uniform(2000, 4095)
            t2 = t1 * random.uniform(0.7, 1.0)   # similar intensities
        elif cls == "DEGRADED":
            vis = random.uniform(0.15, 0.65)
            t1 = random.uniform(1000, 3500)
            t2 = t1 * random.uniform(0.3, 0.7)   # moderate mismatch
        else:  # LOST
            vis = random.uniform(0.0, 0.25)
            t1 = random.uniform(0, 2000)
            t2 = t1 * random.uniform(0.0, 0.3)   # large mismatch or both low

        phase = random.uniform(0, 2 * np.pi)
        baseline = random.uniform(200, 500)

        # Add measurement noise
        vis += random.gauss(0, 0.05)
        vis = max(0, min(1, vis))

        X.append([vis, phase, baseline, t1, t2])
        y.append(cls)

    return np.array(X), np.array(y)


def train(n_samples=2000):
    """
    Train the Random Forest on synthetic data.
    Prints classification report and saves model to model.pkl.

    Returns:
        Trained model, accuracy score.
    """
    global _model

    print("[classifier] Generating training data...")
    X, y = generate_training_data(n_samples)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[classifier] Training Random Forest...")
    _model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    _model.fit(X_train, y_train)

    # Evaluate
    accuracy = _model.score(X_test, y_test)
    y_pred = _model.predict(X_test)

    print(f"\n[classifier] Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=CLASSES))

    # Save model
    joblib.dump(_model, MODEL_PATH)
    print(f"[classifier] Model saved → {MODEL_PATH}")

    return _model, accuracy


def load_model():
    """Load the trained model from model.pkl."""
    global _model

    if not os.path.isfile(MODEL_PATH):
        print("[classifier] No saved model found — training new one...")
        train()
        return _model

    _model = joblib.load(MODEL_PATH)
    print(f"[classifier] Model loaded ← {MODEL_PATH}")
    return _model


def classify(visibility, phase_rad, baseline_mm, t1_intensity, t2_intensity):
    """
    Classify signal quality for a single measurement.

    Args:
        visibility:    fringe visibility [0, 1]
        phase_rad:     phase in radians [0, 2π)
        baseline_mm:   T1-T2 distance in mm
        t1_intensity:  T1 photodiode reading
        t2_intensity:  T2 photodiode reading

    Returns:
        Predicted class: "GOOD", "DEGRADED", or "LOST".
    """
    global _model

    if _model is None:
        load_model()

    features = np.array([[visibility, phase_rad, baseline_mm,
                          t1_intensity, t2_intensity]])
    prediction = _model.predict(features)[0]
    return prediction


def classify_batch(star_results):
    """
    Classify signal quality for a list of star result dicts
    (from visibility.compute_batch).

    Returns:
        List of dicts with original data + "signal_quality" field.
    """
    classified = []
    for star in star_results:
        quality = classify(
            star["visibility"],
            star["phase_rad"],
            star["baseline_mm"],
            star["t1_intensity"],
            star["t2_intensity"],
        )
        result = dict(star)
        result["signal_quality"] = quality
        classified.append(result)
    return classified


# ─── Standalone usage ────────────────────────────────────────
if __name__ == "__main__":
    model, acc = train(3000)
    print(f"\nDemo predictions:")
    print(f"  High V:  {classify(0.85, 1.0, 300, 3500, 3200)}")
    print(f"  Mid V:   {classify(0.40, 2.0, 300, 2000, 1200)}")
    print(f"  Low V:   {classify(0.05, 0.5, 300, 500, 50)}")
