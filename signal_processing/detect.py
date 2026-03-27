#!/usr/bin/env python3
"""
Star Detection Module
=====================
Loads a fringe image (real or synthetic), applies thresholding,
and detects bright point sources. Returns a list of detected
star positions with their peak intensity.
"""

import os

import cv2
import numpy as np


# ─── Detection parameters ────────────────────────────────────
DEFAULT_THRESHOLD = 180       # grayscale threshold (0–255)
MIN_STAR_AREA = 4             # minimum contour area in pixels
MAX_STAR_AREA = 500           # maximum contour area (reject large blobs)
GAUSSIAN_BLUR_SIZE = 3        # pre-processing blur kernel


def load_image(path):
    """
    Load an image as grayscale. Accepts PNG, JPEG, etc.
    Returns 8-bit grayscale NumPy array, or None on failure.
    """
    if not os.path.isfile(path):
        print(f"[detect] File not found: {path}")
        return None

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[detect] Failed to load: {path}")
        return None

    return img


def detect_stars(image, threshold=DEFAULT_THRESHOLD):
    """
    Detect bright point sources in a grayscale image.

    Steps:
        1. Gaussian blur to reduce noise
        2. Binary threshold to isolate bright regions
        3. Find contours
        4. Filter by area (reject noise specks and large blobs)
        5. Compute centroid and peak intensity for each detection

    Args:
        image: 8-bit grayscale NumPy array
        threshold: brightness threshold (0–255)

    Returns:
        List of dicts: [{"x": int, "y": int, "intensity": float}, ...]
    """
    if image is None:
        return []

    # Step 1: blur to smooth noise
    blurred = cv2.GaussianBlur(image, (GAUSSIAN_BLUR_SIZE, GAUSSIAN_BLUR_SIZE), 0)

    # Step 2: binary threshold
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

    # Step 3: find contours of bright regions
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 4–5: filter and extract centroids
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_STAR_AREA or area > MAX_STAR_AREA:
            continue

        # Centroid via image moments
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        # Peak intensity at centroid (from original, not blurred)
        intensity = float(image[cy, cx]) / 255.0

        detections.append({
            "x": cx,
            "y": cy,
            "intensity": round(intensity, 4),
        })

    # Sort by intensity (brightest first)
    detections.sort(key=lambda d: d["intensity"], reverse=True)
    return detections


def detect_from_file(path, threshold=DEFAULT_THRESHOLD):
    """
    Convenience: load image from path and detect stars.
    Returns list of detections or empty list on failure.
    """
    image = load_image(path)
    if image is None:
        return []
    return detect_stars(image, threshold)


# ─── Standalone usage ────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python detect.py <image_path> [threshold]")
        sys.exit(1)

    path = sys.argv[1]
    thresh = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_THRESHOLD

    stars = detect_from_file(path, thresh)
    print(f"Detected {len(stars)} stars in {path}:")
    for s in stars:
        print(f"  ({s['x']}, {s['y']}) intensity={s['intensity']}")
